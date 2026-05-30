# src/data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    class_weights: dict[int, float] | None


def load_image_dataset(
    path: str,
    img_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    validation_split: float | None = None,
    subset: str | None = None,
) -> tf.data.Dataset:
    """Lädt ein Bilddataset aus einer Verzeichnisstruktur."""
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
    """Wendet optionales Subsampling, Caching und Prefetching an."""
    if take > 0:
        ds = ds.take(take)
    if cache:
        ds = ds.cache()
    return ds.prefetch(tf.data.AUTOTUNE)


def count_samples(ds: tf.data.Dataset) -> int:
    """Zählt performant die Anzahl aller Samples (nur anhand der Labels)."""
    total = 0
    for _, y in ds:
        total += y.shape[0]
    return total


def count_classes(ds: tf.data.Dataset, num_classes: int) -> np.ndarray:
    """Ermittelt die Klassenverteilung eines Datasets, ohne Bilder zu verarbeiten."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y_batch in ds:
        counts += np.bincount(y_batch.numpy(), minlength=num_classes)
    return counts


def compute_class_weights_from_counts(counts: np.ndarray) -> dict[int, float]:
    """Berechnet balancierte Klassengewichte aus Klassenhäufigkeiten."""
    total = counts.sum()
    num_classes = len(counts)
    return {
        int(i): float(total / (num_classes * counts[i]))
        for i in range(num_classes)
        if counts[i] > 0
    }


def build_datasets(config: PipelineConfig) -> DatasetBundle:
    """
    Lädt Train-, Validierungs- und Testdaten und bereitet sie für die Pipeline auf.
    """
    # 1. Rohe Datasets laden
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
        shuffle=False,
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

    # 2. Falls ein Limit (take) gesetzt ist, die Rohdaten direkt kürzen
    # Dadurch stimmen Counts und Weights exakt mit dem überein, was das Modell sieht
    if config.train_take > 0:
        train_raw = train_raw.take(config.train_take)
    if config.val_take > 0:
        val_raw = val_raw.take(config.val_take)
    if config.test_take > 0:
        test_raw = test_raw.take(config.test_take)

    # 3. Statistiken auf den (potenziell gekürzten) Daten ermitteln
    train_samples = count_samples(train_raw)
    val_samples = count_samples(val_raw)
    test_samples = count_samples(test_raw)

    class_counts = None
    class_weights = None

    if config.use_class_weights:
        class_counts = count_classes(train_raw, num_classes)
        class_weights = compute_class_weights_from_counts(class_counts)

    # 4. Caching & Prefetching als allerletzten Schritt anwenden
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
        class_weights=class_weights,
    )
