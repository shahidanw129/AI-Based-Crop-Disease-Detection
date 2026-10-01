import json

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")
pytest.importorskip("cv2")

from app.services.prediction import ModelUnavailableError, predict_image


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