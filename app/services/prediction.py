import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


class ModelUnavailableError(Exception):
    pass


@lru_cache(maxsize=2)
def _load_model(model_path):
    path = Path(model_path)
    if not path.is_file():
        raise ModelUnavailableError("The trained model is not installed yet. Follow the training guide to enable detection.")
    try:
        from tensorflow.keras.models import load_model

        model = load_model(path)
    except ImportError as error:
        raise ModelUnavailableError("TensorFlow is not installed. Install the optional ML requirements to enable detection.") from error
    except Exception as error:
        raise ModelUnavailableError("The selected model could not be loaded. Check the model file and try again.") from error

    if model.input_shape != (None, 224, 224, 3):
        raise ModelUnavailableError("The selected model must accept 224x224 RGB images.")
    if not isinstance(model.output_shape, tuple) or len(model.output_shape) != 2 or model.output_shape[-1] is None:
        raise ModelUnavailableError("The selected model must output one class-probability vector.")
    return model


def _read_labels(labels_path, model):
    try:
        labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ModelUnavailableError("The model files could not be read. Check the class labels file and try again.") from error
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ModelUnavailableError("The model class labels are invalid. Check the labels file and try again.")
    if len(labels) != model.output_shape[-1]:
        raise ModelUnavailableError("Model classes do not match the label file. Check the selected model and labels.")
    return labels


def validate_model_artifacts(model_path, labels_path):
    model = _load_model(str(Path(model_path)))
    labels = _read_labels(labels_path, model)
    try:
        import tensorflow as tf

        tf.keras.Model(model.inputs, [_last_spatial_tensor(model), model.output])
    except ImportError as error:
        raise ModelUnavailableError("TensorFlow is not installed. Install the optional ML requirements to enable detection.") from error
    except (TypeError, ValueError, ModelUnavailableError) as error:
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
    heatmap = np.uint8(255 * heatmap.numpy())

    with Image.open(source_image_path) as source_image:
        source_image = ImageOps.exif_transpose(source_image).convert("RGB")
        original_pixels = np.asarray(source_image, dtype=np.uint8)
    height, width = original_pixels.shape[:2]
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_pixels, 0.58, colored_heatmap, 0.42, 0)
    Image.fromarray(overlay).save(heatmap_path, format="JPEG", quality=92, optimize=True)


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