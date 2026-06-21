# src/data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import PipelineConfig


@dataclass
class DatasetBundle:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    class_names: list[str]
    num_classes: int
    train_samples: int
    val_samples: int
    test_samples: int
    class_counts: np.ndarray | None
    val_class_counts: np.ndarray | None
    test_class_counts: np.ndarray | None
    class_weights: dict[int, float] | None

    # Optional metadata for patient-level runs
    train_patient_ids: list[str] | None = None
    val_patient_ids: list[str] | None = None
    test_patient_ids: list[str] | None = None

    overlap_train_val: int | None = None
    overlap_train_test: int | None = None
    overlap_val_test: int | None = None

    patient_leakage_checked: bool | None = None
    patient_id_source: str | None = None
    split_description: str | None = None


def load_image_dataset(
    path: str,
    img_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    validation_split: float | None = None,
    subset: str | None = None,
) -> tf.data.Dataset:
    """Load an image dataset from a directory tree using Keras helpers."""
    return keras.utils.image_dataset_from_directory(
        path,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        label_mode="int",
        validation_split=validation_split,
        subset=subset,
    )


def prepare_dataset(ds: tf.data.Dataset, take: int, cache: bool) -> tf.data.Dataset:
    """Apply optional subsampling, caching and prefetching."""
    if take > 0:
        ds = ds.take(take)
    if cache:
        ds = ds.cache()
    return ds.prefetch(tf.data.AUTOTUNE)


def count_samples(ds: tf.data.Dataset) -> int:
    """Count all samples in a dataset."""
    total = 0
    for _, y in ds:
        total += y.shape[0]
    return total


def count_classes(ds: tf.data.Dataset, num_classes: int) -> np.ndarray:
    """Compute class counts from a dataset."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y_batch in ds:
        counts += np.bincount(y_batch.numpy(), minlength=num_classes)
    return counts


def compute_class_weights_from_counts(counts: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights from class frequencies."""
    total = counts.sum()
    num_classes = len(counts)
    return {
        int(i): float(total / (num_classes * counts[i]))
        for i in range(num_classes)
        if counts[i] > 0
    }


def build_datasets(config: PipelineConfig) -> DatasetBundle:
    """
    Standard image-level split based on image_dataset_from_directory.
    This keeps the old behavior for the original training pipeline.
    """
    train_raw = load_image_dataset(
        path=str(config.train_dir),
        img_size=config.img_size,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        validation_split=config.val_split,
        subset="training",
    )

    val_raw = load_image_dataset(
        path=str(config.train_dir),
        img_size=config.img_size,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        validation_split=config.val_split,
        subset="validation",
    )

    test_raw = load_image_dataset(
        path=str(config.test_dir),
        img_size=config.img_size,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    class_names = list(train_raw.class_names)
    num_classes = len(class_names)

    if config.train_take > 0:
        train_raw = train_raw.take(config.train_take)
    if config.val_take > 0:
        val_raw = val_raw.take(config.val_take)
    if config.test_take > 0:
        test_raw = test_raw.take(config.test_take)

    train_samples = count_samples(train_raw)
    val_samples = count_samples(val_raw)
    test_samples = count_samples(test_raw)

    class_counts = count_classes(train_raw, num_classes)
    val_class_counts = count_classes(val_raw, num_classes)
    test_class_counts = count_classes(test_raw, num_classes)

    class_weights = None
    if config.use_class_weights:
        class_weights = compute_class_weights_from_counts(class_counts)

    # train_ds = prepare_dataset(train_raw, take=-1, cache=False)
    train_ds = prepare_dataset(train_raw, take=-1, cache=config.cache)
    val_ds = prepare_dataset(val_raw, take=-1, cache=config.cache)
    test_ds = prepare_dataset(test_raw, take=-1, cache=config.cache)

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        class_names=class_names,
        num_classes=num_classes,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        class_counts=class_counts,
        val_class_counts=val_class_counts,
        test_class_counts=test_class_counts,
        class_weights=class_weights,
        split_description="image-level split using image_dataset_from_directory",
    )


def _list_class_names(root: Path) -> list[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _list_image_files(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files: list[Path] = []

    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in exts:
                files.append(file_path)

    return sorted(files)


def _extract_patient_id(path: Path) -> str:
    """
    Expected filename examples:
    - DME-943690-12.jpeg
    - CNV-7237325-10.jpg
    - DRUSEN-9624303-3.png
    - NORMAL-1839546-2.jpeg

    Patient ID is the middle part -> 943690, 7237325, ...

    Regex:
    ^([A-Za-z]+)-([A-Za-z0-9]+)-(\d+)$
    """
    stem = path.stem
    match = re.match(r"^([A-Za-z]+)-([A-Za-z0-9]+)-(\d+)$", stem)
    if not match:
        raise ValueError(
            f"Could not extract patient_id from filename '{path.name}'. "
            "Expected format like 'DME-943690-12.jpeg'."
        )
    return match.group(2)


def _build_items_from_directory(root: Path, class_names: list[str]) -> list[dict]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    items: list[dict] = []

    for file_path in _list_image_files(root):
        class_name = file_path.parent.name
        if class_name not in class_to_idx:
            continue

        items.append(
            {
                "path": str(file_path),
                "label": class_to_idx[class_name],
                "class_name": class_name,
                "patient_id": _extract_patient_id(file_path),
            }
        )

    return items


def _split_patient_ids(
    patient_ids: list[str],
    val_split: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    patient_ids = sorted(patient_ids)
    rng = np.random.default_rng(seed)

    shuffled = patient_ids.copy()
    rng.shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_split)))
    if val_count >= len(shuffled):
        val_count = max(1, len(shuffled) - 1)

    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])

    return train_ids, val_ids


def _filter_items_by_patients(items: list[dict], patient_ids: set[str]) -> list[dict]:
    return [item for item in items if item["patient_id"] in patient_ids]


def _count_labels_from_items(items: list[dict], num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for item in items:
        counts[item["label"]] += 1
    return counts


def _make_tf_dataset_from_items(
    items: list[dict],
    img_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = [item["path"] for item in items]
    labels = [item["label"] for item in items]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle and len(items) > 0:
        ds = ds.shuffle(
            buffer_size=len(items),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32)
        return image, label

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds


def build_patient_split_datasets(config: PipelineConfig) -> DatasetBundle:
    """
    Patient-level split:
    - train_dir is split into TRAIN and VALIDATION by patient_id
    - test_dir is treated as an independent TEST set
    - no patient may appear in more than one split
    """
    train_root = Path(config.train_dir)
    test_root = Path(config.test_dir)

    class_names = _list_class_names(train_root)
    num_classes = len(class_names)

    train_val_items = _build_items_from_directory(train_root, class_names)
    test_items = _build_items_from_directory(test_root, class_names)

    all_train_val_patient_ids = sorted({item["patient_id"] for item in train_val_items})
    test_patient_ids = sorted({item["patient_id"] for item in test_items})

    train_patient_ids, val_patient_ids = _split_patient_ids(
        patient_ids=all_train_val_patient_ids,
        val_split=config.val_split,
        seed=config.seed,
    )

    train_items = _filter_items_by_patients(train_val_items, set(train_patient_ids))
    val_items = _filter_items_by_patients(train_val_items, set(val_patient_ids))

    overlap_train_val = len(set(train_patient_ids) & set(val_patient_ids))
    overlap_train_test = len(set(train_patient_ids) & set(test_patient_ids))
    overlap_val_test = len(set(val_patient_ids) & set(test_patient_ids))

    if overlap_train_val != 0 or overlap_train_test != 0 or overlap_val_test != 0:
        raise ValueError(
            "Patient leakage detected: train/val/test patient sets are not disjoint."
        )

    if config.train_take > 0:
        train_items = train_items[: config.train_take]
    if config.val_take > 0:
        val_items = val_items[: config.val_take]
    if config.test_take > 0:
        test_items = test_items[: config.test_take]

    train_raw = _make_tf_dataset_from_items(
        items=train_items,
        img_size=config.img_size,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_raw = _make_tf_dataset_from_items(
        items=val_items,
        img_size=config.img_size,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    test_raw = _make_tf_dataset_from_items(
        items=test_items,
        img_size=config.img_size,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    train_samples = len(train_items)
    val_samples = len(val_items)
    test_samples = len(test_items)

    class_counts = _count_labels_from_items(train_items, num_classes)
    val_class_counts = _count_labels_from_items(val_items, num_classes)
    test_class_counts = _count_labels_from_items(test_items, num_classes)

    class_weights = None
    if config.use_class_weights:
        class_weights = compute_class_weights_from_counts(class_counts)

    # train_ds = prepare_dataset(train_raw, take=-1, cache=False)
    train_ds = prepare_dataset(train_raw, take=-1, cache=config.cache)
    val_ds = prepare_dataset(val_raw, take=-1, cache=config.cache)
    test_ds = prepare_dataset(test_raw, take=-1, cache=config.cache)

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        class_names=class_names,
        num_classes=num_classes,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        class_counts=class_counts,
        val_class_counts=val_class_counts,
        test_class_counts=test_class_counts,
        class_weights=class_weights,
        train_patient_ids=train_patient_ids,
        val_patient_ids=val_patient_ids,
        test_patient_ids=test_patient_ids,
        overlap_train_val=overlap_train_val,
        overlap_train_test=overlap_train_test,
        overlap_val_test=overlap_val_test,
        patient_leakage_checked=True,
        patient_id_source="middle token from filename pattern CLASS-PATIENTID-INDEX",
        split_description="patient-level split for train/validation, separate patient-level test set",
    )