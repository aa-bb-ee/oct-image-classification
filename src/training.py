# src/training.py
from __future__ import annotations

from tensorflow import keras

from src.callbacks import create_stage1_callbacks, create_stage2_callbacks
from src.config import PipelineConfig
from src.data_loader import DatasetBundle
from src.helpers import merge_histories, save_json
from src.model import build_model, compile_model, unfreeze_layers
from src.paths import ExperimentPaths


def _build_config_payload(
    config: PipelineConfig,
    data: DatasetBundle,
) -> dict:
    """Erzeugt ein serialisierbares Konfigurations-/Run-Metadaten-Dictionary."""
    payload = config.to_dict()

    payload["class_names"] = data.class_names
    payload["num_classes"] = data.num_classes
    payload["train_samples_used"] = data.train_samples
    payload["val_samples_used"] = data.val_samples
    payload["test_samples_used"] = data.test_samples

    if data.class_counts is not None:
        payload["train_class_counts"] = {
            data.class_names[i]: int(data.class_counts[i])
            for i in range(data.num_classes)
        }

    if data.val_class_counts is not None:
        payload["val_class_counts"] = {
            data.class_names[i]: int(data.val_class_counts[i])
            for i in range(data.num_classes)
        }

    if data.test_class_counts is not None:
        payload["test_class_counts"] = {
            data.class_names[i]: int(data.test_class_counts[i])
            for i in range(data.num_classes)
        }

    if data.class_weights is not None:
        payload["class_weights"] = {
            str(k): float(v) for k, v in data.class_weights.items()
        }

    return payload


def train_model(
    config: PipelineConfig,
    data: DatasetBundle,
    paths: ExperimentPaths,
) -> tuple[keras.Model, dict]:
    """
    Führt das komplette Training durch:
    - Stage 1: Feature Extraction
    - optional Stage 2: Fine-Tuning
    - speichert Historie und Run-Konfiguration
    """
    model, base_model = build_model(config, data.num_classes)

    compile_model(
        model=model,
        learning_rate=config.learning_rate,
        num_classes=data.num_classes,
    )

    h1 = model.fit(
        data.train_ds,
        validation_data=data.val_ds,
        epochs=config.epochs,
        class_weight=data.class_weights,
        callbacks=create_stage1_callbacks(paths),
        verbose=1,
    )

    h2 = None
    if config.fine_tune:
        unfreeze_layers(base_model, config.unfreeze_last_n)

        compile_model(
            model=model,
            learning_rate=config.fine_tune_lr,
            num_classes=data.num_classes,
        )

        h2 = model.fit(
            data.train_ds,
            validation_data=data.val_ds,
            initial_epoch=config.epochs,
            epochs=config.epochs + config.fine_tune_epochs,
            class_weight=data.class_weights,
            callbacks=create_stage2_callbacks(paths),
            verbose=1,
        )

    history_dict = merge_histories(h1, h2)
    config_payload = _build_config_payload(config, data)

    save_json(paths.met_dir / "history.json", history_dict)
    save_json(paths.met_dir / "run_config.json", config_payload)

    return model, history_dict