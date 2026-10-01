import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras


def evaluate(data_root: Path, model_path: Path, labels_path: Path, output_root: Path):
    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    test_path = data_root / "test"
    if not test_path.is_dir():
        raise ValueError(f"Expected held-out test folder at {test_path}. Run prepare_dataset.py first.")
    dataset = keras.utils.image_dataset_from_directory(
        test_path,
        image_size=(224, 224),
        batch_size=32,
        label_mode="int",
        class_names=class_names,
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)
    model = keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(dataset, verbose=1)
    true_values = np.concatenate([labels.numpy() for _, labels in dataset])
    probabilities = model.predict(dataset, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(true_values, predictions, target_names=class_names, output_dict=True, zero_division=0)
    metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy), "classification_report": report}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = confusion_matrix(true_values, predictions)
    figure_size = max(8, min(18, len(class_names) * 0.55))
    plt.figure(figsize=(figure_size, figure_size))
    sns.heatmap(matrix, cmap="YlGn", xticklabels=class_names, yticklabels=class_names, square=True, cbar=False)
    plt.title("Held-out test set confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_root / "confusion_matrix.png", dpi=180)
    plt.close()

    print(f"Held-out test accuracy: {accuracy:.4f}")
    print(f"Saved per-class precision, recall, F1, and confusion matrix under {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model against the independent test split.")
    parser.add_argument("--data", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--model", type=Path, default=Path("models/crop_disease_model.keras"))
    parser.add_argument("--labels", type=Path, default=Path("models/class_names.json"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    evaluate(args.data, args.model, args.labels, args.output)


if __name__ == "__main__":
    main()