import shutil

import pytest
from PIL import Image

pytest.importorskip("sklearn")

from ai_model.prepare_dataset import split_dataset


def test_split_filters_classes_skips_corrupt_and_exact_duplicates(tmp_path):
    source = tmp_path / "source"
    selected_names = ["Tomato___healthy", "Potato___healthy"]
    for class_index, class_name in enumerate(selected_names):
        class_dir = source / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(7):
            color = (class_index * 60 + image_index * 10, image_index * 12, class_index * 30)
            Image.new("RGB", (24, 24), color=color).save(class_dir / f"leaf-{image_index}.jpg")
    unused_dir = source / "Unused___class"
    unused_dir.mkdir()
    Image.new("RGB", (24, 24), color="purple").save(unused_dir / "unused.jpg")
    (source / selected_names[0] / "broken.jpg").write_bytes(b"not an image")
    shutil.copy2(source / selected_names[0] / "leaf-0.jpg", source / selected_names[0] / "duplicate.jpg")

    summary = split_dataset(source, tmp_path / "splits", selected_classes=selected_names)

    assert set(summary["classes"]) == set(selected_names)
    assert summary["classes"][selected_names[0]]["total"] == 7
    assert summary["skipped_corrupt"] == 1
    assert summary["skipped_exact_duplicates"] == 1
    assert (tmp_path / "splits" / "split_summary.json").is_file()
    assert not (tmp_path / "splits" / "train" / "Unused___class").exists()
    for split in ("train", "validation", "test"):
        assert all((tmp_path / "splits" / split / name).is_dir() for name in selected_names)