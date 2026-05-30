# src/callbacks.py
from __future__ import annotations

from tensorflow import keras

from src.paths import ExperimentPaths


def create_stage1_callbacks(paths: ExperimentPaths) -> list[keras.callbacks.Callback]:
    """
    Erstellt die Callback-Liste für Stage 1 (Feature Extraction).
    """
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(paths.best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=2,
            factor=0.2,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(paths.log_dir),
            histogram_freq=0,
            write_graph=True,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def create_stage2_callbacks(paths: ExperimentPaths) -> list[keras.callbacks.Callback]:
    """
    Erstellt die Callback-Liste für Stage 2 (Fine-Tuning).
    """
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(paths.best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=3,
            factor=0.2,
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]
