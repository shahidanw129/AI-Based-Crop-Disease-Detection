import gc
import json
import hashlib
import logging
import os
import tempfile
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

# Render RAM Optimization: Matplotlib backend ko Headless Agg par set kiya gaya hai
import matplotlib
matplotlib.use("Agg")

import numpy as np
from PIL import Image, ImageOps


class ModelUnavailableError(Exception):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)
_artifact_download_lock = threading.Lock()
_MAX_MODEL_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_LABEL_ARTIFACT_BYTES = 2 * 1024 * 1024


def resolve_project_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_artifact_path(value, *, url_variable, hash_variable, filename, max_bytes):
    local_path = resolve_project_path(value)
    if local_path.is_file():
        return local_path

    artifact_url = os.getenv(url_variable, "").strip()
    if not artifact_url:
        return local_path
    parsed_url = urlparse(artifact_url)
    if parsed_url.scheme != "https" or not parsed_url.hostname:
        logger.error("Remote model artifact URL in %s must use HTTPS", url_variable)
        raise ModelUnavailableError("The selected model artifact is unavailable. Contact the administrator.")
    if parsed_url.username or parsed_url.password:
        logger.error("Remote model artifact URL in %s must not embed credentials", url_variable)
        raise ModelUnavailableError("The selected model artifact is unavailable. Contact the administrator.")

    expected_hash = os.getenv(hash_variable, "").strip().lower()
    if os.getenv("VERCEL") == "1" and not expected_hash:
        logger.error("Vercel model artifact source requires %s for integrity validation", hash_variable)
        raise ModelUnavailableError("The selected model artifact cannot be verified. Contact the administrator.")

    cache_root_value = os.getenv("MODEL_CACHE_DIR", "").strip()
    if cache_root_value:
        cache_root = resolve_project_path(cache_root_value)
    elif os.getenv("VERCEL") == "1":
        cache_root = Path(tempfile.gettempdir()) / "fieldnote-model-cache"
    else:
        cache_root = Path(tempfile.gettempdir()) / "fieldnote-model-cache"
    artifact_name = Path(filename)
    cache_key = hashlib.sha256(artifact_url.encode("utf-8")).hexdigest()[:20]
    cache_path = cache_root / f"{artifact_name.stem}-{cache_key}{artifact_name.suffix}"

    with _artifact_download_lock:
        if cache_path.is_file() and cache_path.stat().st_size > 0:
            if not expected_hash:
                return cache_path
            cached_digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            if cached_digest == expected_hash:
                return cache_path
            logger.error("Cached model artifact failed integrity validation; refreshing it")
            cache_path.unlink(missing_ok=True)

        cache_root.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_root / f".{artifact_name.stem}-{uuid.uuid4().hex}.part"
        token = os.getenv("MODEL_ARTIFACT_TOKEN", "").strip()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        byte_count = 0
        digest = hashlib.sha256()
        try:
            import requests

            with requests.get(
                artifact_url,
                headers=headers,
                stream=True,
                timeout=(5, 60),
                allow_redirects=False,
            ) as response:
                if 300 <= response.status_code < 400:
                    logger.error("Remote model artifact endpoint redirected; redirects are disabled")
                    raise ModelUnavailableError("The selected model artifact is unavailable. Contact the administrator.")
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise ModelUnavailableError("The selected model artifact exceeds the configured size limit.")
                with temporary_path.open("wb") as artifact_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        byte_count += len(chunk)
                        if byte_count > max_bytes:
                            raise ModelUnavailableError("The selected model artifact exceeds the configured size limit.")
                        digest.update(chunk)
                        artifact_file.write(chunk)
            if byte_count == 0:
                raise ModelUnavailableError("The selected model artifact is empty.")

            if expected_hash and digest.hexdigest() != expected_hash:
                raise ModelUnavailableError("The selected model artifact failed its integrity check.")
            os.replace(temporary_path, cache_path)
            logger.info(
                "Downloaded model artifact from %s to runtime cache (%d bytes)",
                parsed_url.hostname,
                byte_count,
            )
            return cache_path
        except ModelUnavailableError:
            raise
        except Exception as error:
            logger.error(
                "Could not download model artifact from %s: %s",
                parsed_url.hostname,
                error.__class__.__name__,
            )
            raise ModelUnavailableError("The selected model artifact is unavailable. Contact the administrator.") from error
        finally:
            temporary_path.unlink(missing_ok=True)


# Maxsize ko 1 rakha gaya hai taaki RAM mein 1 se zyada models na rahein
@lru_cache(maxsize=1)
def _load_model(model_path):
    path = _resolve_artifact_path(
        model_path,
        url_variable="MODEL_ARTIFACT_URL",
        hash_variable="MODEL_ARTIFACT_SHA256",
        filename="model.keras",
        max_bytes=_MAX_MODEL_ARTIFACT_BYTES,
    )
    if not path.is_file():
        logger.error("Configured prediction model artifact is missing: %s", path)
        raise ModelUnavailableError("The trained model is not installed yet. Follow the training guide to enable detection.")
    try:
        from tensorflow.keras.models import load_model

        # Keras 3 / Deserialization Mismatch ko bypass karne ke liye compile=False aur safe_mode=False
        try:
            model = load_model(path, compile=False, safe_mode=False)
        except TypeError:
            model = load_model(path, compile=False)
    except ImportError as error:
        logger.error("TensorFlow is unavailable while loading model %s: %s", path, error)
        raise ModelUnavailableError("TensorFlow is not installed. Install the optional ML requirements to enable detection.") from error
    except Exception as error:
        logger.exception("Failed to load configured prediction model %s", path)
        raise ModelUnavailableError("The selected model could not be loaded. Check the model file and try again.") from error

    if model.input_shape != (None, 224, 224, 3):
        raise ModelUnavailableError("The selected model must accept 224x224 RGB images.")
    if not isinstance(model.output_shape, tuple) or len(model.output_shape) != 2 or model.output_shape[-1] is None:
        raise ModelUnavailableError("The selected model must output one class-probability vector.")
    return model


def _read_labels(labels_path, model):
    path = _resolve_artifact_path(
        labels_path,
        url_variable="LABELS_ARTIFACT_URL",
        hash_variable="LABELS_ARTIFACT_SHA256",
        filename="labels.json",
        max_bytes=_MAX_LABEL_ARTIFACT_BYTES,
    )
    try:
        labels = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        logger.error("Could not read prediction class labels at %s: %s", path, error)
        raise ModelUnavailableError("The model files could not be read. Check the class labels file and try again.") from error
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) and label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        logger.error("Prediction class labels are empty, duplicated, or invalid: %s", path)
        raise ModelUnavailableError("The model class labels are invalid. Check the labels file and try again.")
    if len(labels) != model.output_shape[-1]:
        logger.error(
            "Prediction class count mismatch for %s: labels=%d model_outputs=%d",
            path,
            len(labels),
            model.output_shape[-1],
        )
        raise ModelUnavailableError("Model classes do not match the label file. Check the selected model and labels.")
    return labels


def validate_model_artifacts(model_path, labels_path):
    model = _load_model(str(Path(model_path)))
    labels = _read_labels(labels_path, model)
    try:
        import tensorflow as tf

        tf.keras.Model(model.inputs, [_last_spatial_tensor(model), model.output])
    except ImportError as error:
        logger.error("TensorFlow is unavailable for model artifact validation: %s", error)
        raise ModelUnavailableError("TensorFlow is not installed. Install the optional ML requirements to enable detection.") from error
    except (TypeError, ValueError, ModelUnavailableError) as error:
        logger.error("Prediction/Grad-CAM artifact validation failed for %s: %s", model_path, error)
        raise ModelUnavailableError("The selected model is not compatible with the prediction and Grad-CAM workflow.") from error
    return labels


def _last_spatial_tensor(model):
    for layer in reversed(model.layers):
        try:
            output_tensor = layer.output
            output_shape = output_tensor.shape
            input_tensor = layer.input
            input_shape = input_tensor.shape
        except (AttributeError, ValueError):
            continue
        if len(output_shape) == 4 and layer.__class__.__name__ != "InputLayer":
            if layer.__class__.__name__ not in {"Functional", "Sequential"}:
                return output_tensor
        if len(input_shape) == 4 and len(output_shape) != 4:
            return input_tensor
    raise ModelUnavailableError(
        "This model does not expose a spatial CNN feature map for Grad-CAM. Retrain with the project's CNN or MobileNetV2 pipeline."
    )


def _write_gradcam(model, pixels, class_index, source_image_path, heatmap_path):
    try:
        import cv2
        import tensorflow as tf
    except ImportError as error:
        raise ModelUnavailableError("Install the optional ML requirements, including OpenCV, to generate Grad-CAM explanations.") from error

    target_tensor = _last_spatial_tensor(model)
    try:
        grad_model = tf.keras.Model(model.inputs, [target_tensor, model.output])
    except (TypeError, ValueError) as error:
        raise ModelUnavailableError("Grad-CAM is not compatible with this selected model.") from error
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(pixels, axis=0))
    with tf.GradientTape() as tape:
        activations, probabilities = grad_model(input_tensor, training=False)
        class_score = probabilities[:, class_index]
    gradients = tape.gradient(class_score, activations)
    if gradients is None:
        raise ModelUnavailableError("Grad-CAM gradients could not be calculated for this model.")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(channel_weights * activations, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    if float(maximum.numpy()) > 0:
        heatmap = heatmap / maximum
    heatmap_np = np.uint8(255 * heatmap.numpy())

    # Heavy TensorFlow variables ko memory se clear karna
    del grad_model, input_tensor, activations, probabilities, gradients, channel_weights, heatmap
    gc.collect()

    with Image.open(source_image_path) as source_image:
        source_image = ImageOps.exif_transpose(source_image).convert("RGB")
        original_pixels = np.asarray(source_image, dtype=np.uint8)
        
    height, width = original_pixels.shape[:2]
    heatmap_resized = cv2.resize(heatmap_np, (width, height), interpolation=cv2.INTER_CUBIC)
    colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_pixels, 0.58, colored_heatmap, 0.42, 0)
    Image.fromarray(overlay).save(heatmap_path, format="JPEG", quality=92, optimize=True)

    # Clean array memory
    del original_pixels, heatmap_np, heatmap_resized, colored_heatmap, overlay
    gc.collect()


def predict_image(image_path, model_path, labels_path, heatmap_path=None):
    model = _load_model(str(model_path))
    try:
        labels = _read_labels(labels_path, model)
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
            pixels = np.asarray(image, dtype=np.float32)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, ModelUnavailableError) as error:
        if isinstance(error, ModelUnavailableError):
            raise
        raise ModelUnavailableError("The model files could not be read. Retrain the model and try again.") from error

    try:
        probabilities = np.asarray(model(np.expand_dims(pixels, axis=0), training=False))[0]
    except Exception as error:
        raise ModelUnavailableError("The selected model could not process this image.") from error

    best_index = int(np.argmax(probabilities))
    if heatmap_path is None:
        heatmap_path = str(Path(image_path).with_name(f"{Path(image_path).stem}_gradcam.jpg"))
    
    _write_gradcam(model, pixels, best_index, image_path, heatmap_path)
    
    # Garbage Collector call to keep RAM low
    gc.collect()

    raw_label = labels[best_index]
    parts = raw_label.replace("___", "|").replace("__", "|").split("|", 1)
    crop = parts[0].replace("_", " ").title()
    disease = parts[-1].replace("_", " ").title()
    return {
        "raw_label": raw_label,
        "crop_name": crop,
        "disease_name": disease,
        "confidence": float(probabilities[best_index]),
        "explanation_path": str(heatmap_path),
    }