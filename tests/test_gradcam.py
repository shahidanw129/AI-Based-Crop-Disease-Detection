import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")
pytest.importorskip("cv2")

from app.services.prediction import ModelUnavailableError, predict_image, resolve_project_path, validate_model_artifacts


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