import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")
pytest.importorskip("cv2")

from app.services.prediction import (
    ModelUnavailableError,
    predict_image,
    resolve_project_path,
    validate_model_artifacts,
)


def test_prediction_writes_gradcam_overlay(tmp_path):
    image_path = tmp_path / "leaf.jpg"
    heatmap_path = tmp_path / "leaf_gradcam.jpg"
    model_path = tmp_path / "tiny_cnn.keras"
    labels_path = tmp_path / "class_names.json"

    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="last_conv")(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(features)
    tf.keras.Model(inputs, outputs).save(model_path)
    labels_path.write_text(json.dumps(["Tomato___healthy", "Tomato___Early_blight"]), encoding="utf-8")
    Image.fromarray(np.full((70, 90, 3), (42, 126, 57), dtype=np.uint8)).save(image_path)

    result = predict_image(image_path, model_path, labels_path, heatmap_path=heatmap_path)

    assert result["raw_label"] in {"Tomato___healthy", "Tomato___Early_blight"}
    assert result["explanation_path"] == str(heatmap_path)
    assert heatmap_path.is_file()
    with Image.open(heatmap_path) as overlay:
        assert overlay.size == (90, 70)
        assert overlay.format == "JPEG"


def test_prediction_writes_gradcam_for_nested_backbone(tmp_path):
    image_path = tmp_path / "leaf.jpg"
    heatmap_path = tmp_path / "leaf_gradcam.jpg"
    model_path = tmp_path / "nested_cnn.keras"
    labels_path = tmp_path / "class_names.json"

    backbone_inputs = tf.keras.Input(shape=(224, 224, 3))
    features = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="backbone_conv")(backbone_inputs)
    backbone = tf.keras.Model(backbone_inputs, features, name="nested_backbone")
    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = backbone(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(features)
    tf.keras.Model(inputs, outputs).save(model_path)
    labels_path.write_text(json.dumps(["Tomato___healthy", "Tomato___Early_blight"]), encoding="utf-8")
    Image.fromarray(np.full((70, 90, 3), (42, 126, 57), dtype=np.uint8)).save(image_path)

    result = predict_image(image_path, model_path, labels_path, heatmap_path=heatmap_path)

    assert result["raw_label"] in {"Tomato___healthy", "Tomato___Early_blight"}
    assert heatmap_path.is_file()


def test_prediction_rejects_incompatible_input_shape(tmp_path):
    model_path = tmp_path / "wrong_shape.keras"
    inputs = tf.keras.Input(shape=(128, 128, 3))
    features = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(features)
    tf.keras.Model(inputs, outputs).save(model_path)

    with pytest.raises(ModelUnavailableError, match="224x224 RGB"):
        predict_image(
            tmp_path / "missing.jpg",
            model_path,
            tmp_path / "missing-labels.json",
            heatmap_path=tmp_path / "heatmap.jpg",
        )


def test_relative_model_and_labels_resolve_from_project_root():
    project_root = Path(__file__).resolve().parents[1]
    model_path = resolve_project_path("models/crop_disease_model.keras")
    labels_path = resolve_project_path("models/class_names.json")

    assert model_path == (project_root / "models" / "crop_disease_model.keras").resolve()
    assert labels_path == (project_root / "models" / "class_names.json").resolve()
    assert len(validate_model_artifacts("models/crop_disease_model.keras", "models/class_names.json")) == 9


def test_prediction_rejects_label_count_mismatch_before_opening_image(tmp_path):
    model_path = tmp_path / "two_classes.keras"
    labels_path = tmp_path / "class_names.json"
    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(features)
    tf.keras.Model(inputs, outputs).save(model_path)
    labels_path.write_text(json.dumps(["Tomato___healthy"]), encoding="utf-8")

    with pytest.raises(ModelUnavailableError, match="do not match the label file"):
        predict_image(
            tmp_path / "missing.jpg",
            model_path,
            labels_path,
            heatmap_path=tmp_path / "heatmap.jpg",
        )


def test_prediction_downloads_private_model_and_labels_once(tmp_path, monkeypatch):
    import requests

    model_source = tmp_path / "remote_source.keras"
    labels_source = tmp_path / "remote_labels.json"
    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(features)
    tf.keras.Model(inputs, outputs).save(model_source)
    labels_source.write_text(
        json.dumps(["Tomato___healthy", "Tomato___Early_blight"]),
        encoding="utf-8",
    )
    model_bytes = model_source.read_bytes()
    labels_bytes = labels_source.read_bytes()
    artifacts = {
        "https://models.example.test/candidate.keras": model_bytes,
        "https://models.example.test/classes.json": labels_bytes,
    }
    requested = []

    class Response:
        def __init__(self, body):
            self.body = body
            self.status_code = 200
            self.headers = {"Content-Length": str(len(body))}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            for offset in range(0, len(self.body), chunk_size):
                yield self.body[offset:offset + chunk_size]

    def fake_get(url, **kwargs):
        requested.append((url, kwargs))
        return Response(artifacts[url])

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setenv("MODEL_ARTIFACT_URL", "https://models.example.test/candidate.keras")
    monkeypatch.setenv("LABELS_ARTIFACT_URL", "https://models.example.test/classes.json")
    monkeypatch.setenv("MODEL_ARTIFACT_TOKEN", "test-only-token")
    monkeypatch.setenv("MODEL_ARTIFACT_SHA256", hashlib.sha256(model_bytes).hexdigest())
    monkeypatch.setenv("LABELS_ARTIFACT_SHA256", hashlib.sha256(labels_bytes).hexdigest())
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "runtime-cache"))

    image_path = tmp_path / "leaf.jpg"
    heatmap_path = tmp_path / "leaf-gradcam.jpg"
    Image.fromarray(np.full((64, 80, 3), (45, 120, 55), dtype=np.uint8)).save(image_path)
    model_path = tmp_path / "deployment-model.keras"
    labels_path = tmp_path / "deployment-labels.json"

    result = predict_image(image_path, model_path, labels_path, heatmap_path)

    assert result["raw_label"] in {"Tomato___healthy", "Tomato___Early_blight"}
    assert heatmap_path.is_file()
    assert len(requested) == 2
    assert all(call[1]["headers"]["Authorization"] == "Bearer test-only-token" for call in requested)


def test_prediction_rejects_duplicate_remote_or_local_labels(tmp_path):
    model_path = tmp_path / "duplicate-label-model.keras"
    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(features)
    tf.keras.Model(inputs, outputs).save(model_path)
    labels_path = tmp_path / "duplicate-labels.json"
    labels_path.write_text(json.dumps(["Tomato___healthy", "Tomato___healthy"]), encoding="utf-8")

    with pytest.raises(ModelUnavailableError, match="class labels are invalid"):
        predict_image(
            tmp_path / "missing.jpg",
            model_path,
            labels_path,
            tmp_path / "heatmap.jpg",
        )


def test_private_remote_model_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    import requests

    class Response:
        status_code = 200
        headers = {"Content-Length": "11"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"not a model"

    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: Response())
    monkeypatch.setenv("MODEL_ARTIFACT_URL", "https://models.example.test/model.keras")
    monkeypatch.setenv("MODEL_ARTIFACT_SHA256", "0" * 64)
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(ModelUnavailableError, match="integrity check"):
        predict_image(
            tmp_path / "missing-image.jpg",
            tmp_path / "missing-model.keras",
            tmp_path / "missing-labels.json",
        )


def test_vercel_remote_artifact_requires_checksum(tmp_path, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("MODEL_ARTIFACT_URL", "https://models.example.test/model.keras")
    monkeypatch.delenv("MODEL_ARTIFACT_SHA256", raising=False)
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(ModelUnavailableError, match="cannot be verified"):
        predict_image(
            tmp_path / "missing-image.jpg",
            tmp_path / "missing-model.keras",
            tmp_path / "missing-labels.json",
        )


def test_private_remote_artifact_redirect_is_rejected(tmp_path, monkeypatch):
    import requests

    class RedirectResponse:
        status_code = 302
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            return iter(())

    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: RedirectResponse())
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("MODEL_ARTIFACT_URL", "https://models.example.test/model.keras")
    monkeypatch.delenv("MODEL_ARTIFACT_SHA256", raising=False)
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(ModelUnavailableError, match="artifact is unavailable"):
        predict_image(
            tmp_path / "missing-image.jpg",
            tmp_path / "missing-model.keras",
            tmp_path / "missing-labels.json",
        )