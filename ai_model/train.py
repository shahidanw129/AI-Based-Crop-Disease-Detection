import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model(class_count: int, architecture: str):
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="leaf_rgb")
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.12),
        ],
        name="training_augmentation",
    )
    images = augmentation(inputs)

    if architecture == "mobilenetv2":
        images = layers.Rescaling(1.0 / 127.5, offset=-1)(images)
        backbone = keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
        )
        backbone.trainable = False
        features = backbone(images, training=False)
        features = layers.GlobalAveragePooling2D()(features)
        features = layers.Dropout(0.25)(features)
    else:
        images = layers.Rescaling(1.0 / 255)(images)
        features = images
        for filters in (32, 64, 128, 256):
            features = layers.Conv2D(filters, 3, padding="same", use_bias=False)(features)
            features = layers.BatchNormalization()(features)
            features = layers.Activation("relu")(features)
            features = layers.MaxPooling2D()(features)
            if filters >= 128:
                features = layers.Dropout(0.15)(features)
        features = layers.GlobalAveragePooling2D()(features)
        features = layers.Dropout(0.3)(features)

    outputs = layers.Dense(class_count, activation="softmax", name="class_probabilities")(features)
    model = keras.Model(inputs, outputs, name=f"crop_disease_{architecture}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_split(path: Path, class_names=None, shuffle=False):
    return keras.utils.image_dataset_from_directory(
        path,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=class_names,
        shuffle=shuffle,
        seed=42,
    )


def train(data_root: Path, output_root: Path, architecture="cnn", epochs=25, initial_model=None):
    train_path = data_root / "train"
    validation_path = data_root / "validation"
    if not train_path.is_dir() or not validation_path.is_dir():
        raise ValueError(f"Expected {train_path} and {validation_path}. Run prepare_dataset.py first.")

    train_data = load_split(train_path, shuffle=True)
    class_names = train_data.class_names
    if len(class_names) < 2:
        raise ValueError("The training split must contain at least two class folders.")
    validation_data = load_split(validation_path, class_names=class_names)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

    class_counts = [
        sum(1 for path in (train_path / class_name).rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        for class_name in class_names
    ]
    if any(count == 0 for count in class_counts):
        raise ValueError("Every class needs at least one training image.")
    total_train_images = sum(class_counts)
    class_weights = {
        index: total_train_images / (len(class_counts) * count)
        for index, count in enumerate(class_counts)
    }

    output_root.mkdir(parents=True, exist_ok=True)
    if initial_model:
        model = keras.models.load_model(initial_model)
        if model.output_shape[-1] != len(class_names):
            raise ValueError("Resume checkpoint output classes do not match the training folders.")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=2e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model = build_model(len(class_names), architecture)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(output_root / "best_model.keras", monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    model.save(output_root / "crop_disease_model.keras")
    (output_root / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    metrics = {key: [float(value) for value in values] for key, values in history.history.items()}
    (output_root / "training_history.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        axis.plot(metrics.get(metric, []), label=f"Training {metric}")
        axis.plot(metrics.get(f"val_{metric}", []), label=f"Validation {metric}")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "training_curves.png", dpi=180)
    plt.close(figure)
    print(f"Saved model and {len(class_names)} class names under {output_root}")
    print(f"Applied class weights: {dict(zip(class_names, (round(class_weights[index], 3) for index in range(len(class_names)))))}")
    print("Evaluate the held-out test split before presenting any performance claim.")


def main():
    parser = argparse.ArgumentParser(description="Train a custom CNN or frozen MobileNetV2 transfer-learning baseline.")
    parser.add_argument("--data", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--architecture", choices=("cnn", "mobilenetv2"), default="cnn")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--initial-model", type=Path, help="Continue training from a saved .keras checkpoint.")
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 200:
        parser.error("--epochs must be between 1 and 200")
    train(args.data, args.output, args.architecture, args.epochs, args.initial_model)


if __name__ == "__main__":
    main()