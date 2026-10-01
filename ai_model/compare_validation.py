import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow import keras


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
REPORT_NAME = "validation_comparison.json"


def _load_dataset(data_root: Path, class_names: list[str]):
    validation_path = data_root / "validation"
    if not validation_path.is_dir():
        raise ValueError(f"Expected validation split at {validation_path}.")
    return keras.utils.image_dataset_from_directory(
        validation_path,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=class_names,
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)


def _measure(model_path: Path, dataset, class_names: list[str]):
    model = keras.models.load_model(model_path)
    if model.input_shape != (None, *IMAGE_SIZE, 3):
        raise ValueError(f"Expected RGB {IMAGE_SIZE} input for {model_path}; got {model.input_shape}.")
    if model.output_shape[-1] != len(class_names):
        raise ValueError(f"Output class count for {model_path} does not match the labels.")

    loss, accuracy = model.evaluate(dataset, verbose=0)
    true_values = np.concatenate([labels.numpy() for _, labels in dataset])
    predictions = np.argmax(model.predict(dataset, verbose=0), axis=1)
    report = classification_report(
        true_values,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "validation_loss": float(loss),
        "validation_accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
    }


def compare_validation(
    data_root: Path,
    baseline_path: Path,
    candidate_path: Path,
    labels_path: Path,
    candidate_labels_path: Path,
    output_root: Path,
):
    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    candidate_class_names = json.loads(candidate_labels_path.read_text(encoding="utf-8"))
    if class_names != candidate_class_names:
        raise ValueError("Baseline and candidate class labels or order do not match.")

    report_path = output_root / REPORT_NAME
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")

    dataset = _load_dataset(data_root, class_names)
    baseline = _measure(baseline_path, dataset, class_names)
    candidate = _measure(candidate_path, dataset, class_names)
    metric_names = (
        "validation_loss",
        "validation_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )
    comparison = {
        "split": "validation",
        "sample_count": sum(item["support"] for item in candidate["per_class"].values()),
        "baseline_model": str(baseline_path),
        "candidate_model": str(candidate_path),
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": {
            metric: candidate[metric] - baseline[metric] for metric in metric_names
        },
        "per_class_recall_delta": {
            name: candidate["per_class"][name]["recall"] - baseline["per_class"][name]["recall"]
            for name in class_names
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Validation-only comparison saved to {report_path}")
    print(json.dumps(comparison["candidate_minus_baseline"], indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Compare a candidate checkpoint with a baseline using validation data only."
    )
    parser.add_argument("--data", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidate-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare_validation(
        args.data,
        args.baseline,
        args.candidate,
        args.labels,
        args.candidate_labels,
        args.output,
    )


if __name__ == "__main__":
    main()