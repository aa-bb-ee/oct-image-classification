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


def _history_to_dict(history_obj) -> dict[str, list[float]]:
    """
    NEU:
    Wandelt ein einzelnes Keras-History-Objekt in ein Dictionary um.

    Warum?
    - stage1   erzeugt nur h1
    - finetune erzeugt nur h2
    - full     erzeugt h1 + h2

    merge_histories() ist nur sinnvoll, wenn wirklich zwei Histories
    zusammengeführt werden müssen.
    """
    if hasattr(history_obj, "history"):
        history_obj = history_obj.history

    if not isinstance(history_obj, dict):
        raise TypeError("History must be a Keras History object or a dict.")

    return {k: list(v) for k, v in history_obj.items()}


def _find_backbone_in_loaded_model(model: keras.Model) -> keras.Model:
    """
    NEU:
    Sucht in einem geladenen Gesamtmodell das eingebettete Backbone,
    z.B. InceptionV3, ResNet50, EfficientNet usw.

    Warum?
    Für Fine-Tuning müssen gezielt die letzten Layer des Backbones
    entfroren werden. Wenn ein Modell aus best_model.keras geladen wird,
    haben wir zunächst nur das Gesamtmodell und müssen das Backbone
    darin wiederfinden.

    Voraussetzung:
    Das Modell wurde mit unserer Pipeline gebaut, also über build_model().
    """
    nested_models = [
        layer for layer in model.layers
        if isinstance(layer, keras.Model)
    ]

    if not nested_models:
        raise ValueError(
            "Could not find a nested backbone model in the loaded model. "
            "The model must be saved from this pipeline."
        )

    # In unserer Architektur ist das erste verschachtelte Keras-Modell
    # das Backbone.
    return nested_models[0]


def _run_stage1(
    config: PipelineConfig,
    data: DatasetBundle,
    paths: ExperimentPaths,
) -> tuple[keras.Model, keras.Model, keras.callbacks.History]:
    """
    NEU ausgelagert:
    Führt Stage 1 aus.

    Stage 1 bedeutet:
    - neues Modell bauen
    - Backbone bleibt eingefroren
    - nur der Classification Head wird trainiert
    - Learning Rate: config.learning_rate
    - Epochs: config.epochs
    """
    model, base_model = build_model(config, data.num_classes)

    compile_model(
        model=model,
        learning_rate=config.learning_rate,
        num_classes=data.num_classes,
    )

    history = model.fit(
        data.train_ds,
        validation_data=data.val_ds,
        epochs=config.epochs,
        class_weight=data.class_weights,
        callbacks=create_stage1_callbacks(paths),
        verbose=1,
    )

    return model, base_model, history


def _run_stage2(
    model: keras.Model,
    base_model: keras.Model,
    config: PipelineConfig,
    data: DatasetBundle,
    paths: ExperimentPaths,
    initial_epoch: int,
    final_epoch: int,
) -> keras.callbacks.History:
    """
    NEU ausgelagert:
    Führt Stage 2 / Fine-Tuning aus.

    Stage 2 bedeutet:
    - vorhandenes Modell weiterverwenden
    - letzte N Layer des Backbones entfrieren
    - BatchNorm-Layer bleiben durch unfreeze_layers() eingefroren
    - kleinere Learning Rate: config.fine_tune_lr
    - Epochs werden über initial_epoch/final_epoch gesteuert

    Wird verwendet für:
    - train_mode='full'     nach Stage 1
    - train_mode='finetune' nach Laden eines bestehenden Modells
    """
    unfreeze_layers(base_model, config.unfreeze_last_n)

    compile_model(
        model=model,
        learning_rate=config.fine_tune_lr,
        num_classes=data.num_classes,
    )

    history = model.fit(
        data.train_ds,
        validation_data=data.val_ds,
        initial_epoch=initial_epoch,
        epochs=final_epoch,
        class_weight=data.class_weights,
        callbacks=create_stage2_callbacks(paths),
        verbose=1,
    )

    return history


def train_model(
    config: PipelineConfig,
    data: DatasetBundle,
    paths: ExperimentPaths,
) -> tuple[keras.Model, dict]:
    """
    ZENTRAL GEÄNDERT:
    train_model() steuert jetzt alle Trainingsvarianten über config.train_mode.

    Unterstützte Modi:

    train_mode='stage1'
        Nur Stage 1 / Feature Extraction.
        Das entspricht eurer bisherigen frozen-Variante.

    train_mode='full'
        Stage 1 + Stage 2 in einem einzigen Lauf.
        Das entspricht eurer bisherigen Fine-Tuning-Variante, z.B. ft50.

    train_mode='finetune'
        Ein bereits gespeichertes Modell wird über config.base_model_path
        geladen und nur Stage 2 wird ausgeführt.
        Das ersetzt euer früheres separates start_finetune.sh.
    """
    h1 = None
    h2 = None

    if config.train_mode == "stage1":
        # NEU:
        # Nur Stage 1 ausführen.
        # Kein Fine-Tuning danach.
        model, _, h1 = _run_stage1(
            config=config,
            data=data,
            paths=paths,
        )

    elif config.train_mode == "full":
        # NEU:
        # Zuerst Stage 1 ausführen.
        model, base_model, h1 = _run_stage1(
            config=config,
            data=data,
            paths=paths,
        )

        # Danach direkt Stage 2 / Fine-Tuning ausführen.
        #
        # initial_epoch=config.epochs sorgt dafür, dass die Epoch-Zählung
        # nahtlos weiterläuft:
        # Beispiel: Stage 1 = 20 Epochs, Stage 2 = 10 Epochs
        # Dann läuft Stage 2 von Epoch 20 bis 29.
        h2 = _run_stage2(
            model=model,
            base_model=base_model,
            config=config,
            data=data,
            paths=paths,
            initial_epoch=config.epochs,
            final_epoch=config.epochs + config.fine_tune_epochs,
        )

    elif config.train_mode == "finetune":
        # NEU:
        # Fine-Tune-only-Modus.
        # Hier wird KEIN neues Modell gebaut und KEINE Stage 1 ausgeführt.
        # Stattdessen wird ein vorhandenes Stage-1-Modell geladen.
        if config.base_model_path is None:
            raise ValueError(
                "base_model_path is required when train_mode='finetune'."
            )

        model = keras.models.load_model(
            config.base_model_path,
            compile=False,
        )

        # Das Backbone muss aus dem geladenen Gesamtmodell extrahiert werden,
        # damit unfreeze_layers() darauf angewendet werden kann.
        base_model = _find_backbone_in_loaded_model(model)

        # Bei Fine-Tune-only starten wir die Epoch-Zählung wieder bei 0,
        # weil dies ein neuer separater Run ist.
        h2 = _run_stage2(
            model=model,
            base_model=base_model,
            config=config,
            data=data,
            paths=paths,
            initial_epoch=0,
            final_epoch=config.fine_tune_epochs,
        )

    else:
        raise ValueError(
            "train_mode must be one of: 'stage1', 'full', 'finetune'."
        )

    # NEU:
    # History abhängig vom Modus speichern.
    #
    # stage1   -> nur h1
    # full     -> h1 + h2 zusammenführen
    # finetune -> nur h2
    if h1 is not None and h2 is not None:
        history_dict = merge_histories(h1, h2)
    elif h1 is not None:
        history_dict = _history_to_dict(h1)
    elif h2 is not None:
        history_dict = _history_to_dict(h2)
    else:
        raise RuntimeError("No training history was created.")

    config_payload = _build_config_payload(config, data)

    save_json(paths.met_dir / "history.json", history_dict)
    save_json(paths.met_dir / "run_config.json", config_payload)

    return model, history_dict