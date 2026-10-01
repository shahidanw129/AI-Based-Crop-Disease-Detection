import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def split_dataset(source: Path, output: Path, seed: int = 42, selected_classes=None):
    if not source.is_dir():
        raise ValueError(f"Dataset folder not found: {source}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output folder is not empty: {output}. Choose a new folder for this split.")

    available_classes = {folder.name: folder for folder in source.iterdir() if folder.is_dir()}
    if selected_classes:
        missing = sorted(set(selected_classes) - available_classes.keys())
        if missing:
            raise ValueError(f"Requested classes not found under {source}: {', '.join(missing)}")
        classes = [available_classes[name] for name in selected_classes]
    else:
        classes = sorted(available_classes.values())
    if len(classes) < 2:
        raise ValueError("Provide at least two class folders inside the raw dataset folder.")

    output.mkdir(parents=True, exist_ok=True)
    seen_hashes = set()
    counts = {"train": 0, "validation": 0, "test": 0}
    per_class = {}
    skipped_corrupt = 0
    skipped_duplicates = 0
    for class_folder in classes:
        images = []
        for path in sorted(class_folder.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not _valid_image(path):
                skipped_corrupt += 1
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                skipped_duplicates += 1
                continue
            seen_hashes.add(digest)
            images.append(path)
        if len(images) < 5:
            raise ValueError(f"Class '{class_folder.name}' needs at least 5 unique images; found {len(images)}.")

        train_validation, test = train_test_split(images, test_size=0.15, random_state=seed)
        train, validation = train_test_split(train_validation, test_size=0.1765, random_state=seed)
        for split_name, split_images in (("train", train), ("validation", validation), ("test", test)):
            destination = output / split_name / class_folder.name
            destination.mkdir(parents=True, exist_ok=True)
            for index, image_path in enumerate(split_images):
                target = destination / f"{index:05d}{image_path.suffix.lower()}"
                shutil.copy2(image_path, target)
                counts[split_name] += 1
        per_class[class_folder.name] = {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "total": len(images),
        }

    summary = {
        "source": str(source),
        "seed": seed,
        "classes": per_class,
        "totals": counts,
        "skipped_corrupt": skipped_corrupt,
        "skipped_exact_duplicates": skipped_duplicates,
    }
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "split_counts.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", "train", "validation", "test", "total"])
        for class_name, class_counts in per_class.items():
            writer.writerow([class_name, class_counts["train"], class_counts["validation"], class_counts["test"], class_counts["total"]])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Make non-overlapping train, validation, and test folders.")
    parser.add_argument("--source", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--output", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", help="Optional exact class-folder names to include.")
    args = parser.parse_args()
    summary = split_dataset(args.source, args.output, args.seed, args.classes)
    print(f"Created split totals: {summary['totals']}")
    print(f"Classes: {len(summary['classes'])}; corrupt files skipped: {summary['skipped_corrupt']}; exact duplicates skipped: {summary['skipped_exact_duplicates']}")
    print("Review class balance and remove near-duplicate or field-correlated images before training.")


if __name__ == "__main__":
    main()