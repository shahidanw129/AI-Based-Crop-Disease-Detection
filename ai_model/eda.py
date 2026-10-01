import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from ai_model.prepare_dataset import IMAGE_EXTENSIONS


def analyze_dataset(source: Path, output: Path, selected_classes=None, samples_per_class=4):
    if not source.is_dir():
        raise ValueError(f"Dataset folder not found: {source}")
    available_classes = {folder.name: folder for folder in source.iterdir() if folder.is_dir()}
    if selected_classes:
        missing = sorted(set(selected_classes) - available_classes.keys())
        if missing:
            raise ValueError(f"Requested classes not found under {source}: {', '.join(missing)}")
        class_folders = [available_classes[name] for name in selected_classes]
    else:
        class_folders = sorted(available_classes.values())
    if len(class_folders) < 2:
        raise ValueError("EDA needs at least two class folders.")

    output.mkdir(parents=True, exist_ok=True)
    seen_hashes = {}
    class_report = {}
    corrupt_files = []
    duplicate_groups = []
    dimension_counts = Counter()
    mode_counts = Counter()
    samples = {}

    for class_folder in class_folders:
        valid_paths = []
        class_hashes = Counter()
        for path in sorted(class_folder.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions = f"{image.width}x{image.height}"
                    mode = image.mode
            except (UnidentifiedImageError, OSError, ValueError):
                corrupt_files.append(str(path))
                continue

            valid_paths.append(path)
            dimension_counts[dimensions] += 1
            mode_counts[mode] += 1
            class_hashes[digest] += 1
            seen_hashes.setdefault(digest, []).append(str(path))

        samples[class_folder.name] = valid_paths[:samples_per_class]
        class_report[class_folder.name] = {
            "image_files": len(valid_paths),
            "unique_images_within_class": len(class_hashes),
            "exact_duplicate_files_within_class": len(valid_paths) - len(class_hashes),
        }

    for paths in seen_hashes.values():
        if len(paths) > 1:
            duplicate_groups.append(paths)

    total_images = sum(item["image_files"] for item in class_report.values())
    summary = {
        "source": str(source),
        "selected_class_count": len(class_report),
        "valid_image_count": total_images,
        "corrupt_image_count": len(corrupt_files),
        "exact_duplicate_groups_across_selected_classes": len(duplicate_groups),
        "exact_duplicate_file_count": sum(len(group) - 1 for group in duplicate_groups),
        "dimensions": dict(dimension_counts),
        "color_modes": dict(mode_counts),
        "classes": class_report,
        "quality_notes": [
            "Image-level random splits cannot prevent near-duplicate or same-plant leakage; review source collection groups.",
            "Check field backgrounds, capture conditions, and class-label correctness before interpreting test metrics.",
        ],
    }
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "corrupt_files.txt").write_text("\n".join(corrupt_files), encoding="utf-8")
    (output / "exact_duplicate_groups.json").write_text(json.dumps(duplicate_groups, indent=2), encoding="utf-8")

    class_names = list(class_report)
    counts = [class_report[name]["image_files"] for name in class_names]
    figure_height = max(5, len(class_names) * 0.31)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    axis.barh(class_names, counts, color="#557e59")
    axis.invert_yaxis()
    axis.set_xlabel("Valid image files")
    axis.set_title("Selected PlantVillage class distribution")
    figure.tight_layout()
    figure.savefig(output / "class_distribution.png", dpi=170)
    plt.close(figure)

    columns = min(4, samples_per_class)
    rows = len(class_names)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.8, rows * 2.5), squeeze=False)
    for row_index, class_name in enumerate(class_names):
        for column_index in range(columns):
            axis = axes[row_index][column_index]
            axis.axis("off")
            class_samples = samples[class_name]
            if column_index >= len(class_samples):
                continue
            with Image.open(class_samples[column_index]) as image:
                axis.imshow(image.convert("RGB"))
            if column_index == 0:
                axis.set_ylabel(class_name, fontsize=7)
    figure.suptitle("Sample leaf images by selected class", fontsize=13)
    figure.tight_layout()
    figure.savefig(output / "sample_grid.png", dpi=150)
    plt.close(figure)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit class counts, image integrity, duplicates, dimensions, and sample images.")
    parser.add_argument("--source", type=Path, default=Path("dataset/raw/plantvillage dataset/color"))
    parser.add_argument("--output", type=Path, default=Path("dataset/analysis"))
    parser.add_argument("--classes", nargs="+", help="Optional exact class-folder names to include.")
    args = parser.parse_args()
    summary = analyze_dataset(args.source, args.output, args.classes)
    print(f"Valid images: {summary['valid_image_count']} across {summary['selected_class_count']} classes")
    print(f"Corrupt: {summary['corrupt_image_count']}; exact duplicate groups: {summary['exact_duplicate_groups_across_selected_classes']}")
    print(f"EDA artifacts saved under {args.output}")


if __name__ == "__main__":
    main()