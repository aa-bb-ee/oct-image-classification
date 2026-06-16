# src/model.py
from __future__ import annotations

from collections.abc import Callable

from tensorflow import keras
from tensorflow.keras import layers

from src.config import PipelineConfig


def get_metrics(num_classes: int) -> list[keras.metrics.Metric]:
    """Erzeugt die Standardmetriken für das Training."""
    metrics: list[keras.metrics.Metric] = [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    ]

    if num_classes > 2:
        metrics.append(
            keras.metrics.SparseTopKCategoricalAccuracy(
                k=2,
                name="top2_acc",
            )
        )

    return metrics


def build_augmentation_layer(config: PipelineConfig) -> keras.Sequential:
    """Erstellt optional eine Data-Augmentation-Pipeline."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=config.seed),
            layers.RandomRotation(0.05, seed=config.seed),
            layers.RandomZoom(0.10, seed=config.seed),
        ],
        name="data_augmentation",
    )


def get_preprocess_fn(model_name: str) -> Callable:
    """
    Liefert die zur Architektur passende Preprocessing-Funktion zurück.

    Wichtig:
    Jede Keras-Application erwartet ihr eigenes Preprocessing.
    """
    model_name = model_name.lower()

    if model_name == "inceptionv3":
        return keras.applications.inception_v3.preprocess_input

    if model_name == "resnet50":
        return keras.applications.resnet50.preprocess_input

    if model_name in ("efficientnetb0", "efficientnetb3"):
        # Beide nutzen das gleiche EfficientNet-Preprocessing
        return keras.applications.efficientnet.preprocess_input

    raise ValueError(f"Nicht unterstütztes Modell: {model_name}")


def build_backbone(config: PipelineConfig) -> keras.Model:
    """
    Baut das vortrainierte Backbone ohne Classification Head.
    """
    model_name = config.model_name.lower()
    input_shape = (config.img_size, config.img_size, 3)

    if model_name == "inceptionv3":
        base_model = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
        base_model.trainable = False
        return base_model

    if model_name == "resnet50":
        base_model = keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
        base_model.trainable = False
        return base_model

    if model_name == "efficientnetb0":
        base_model = keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
        base_model.trainable = False
        return base_model

    if model_name == "efficientnetb3":
        base_model = keras.applications.EfficientNetB3(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
        base_model.trainable = False
        return base_model

    raise ValueError(f"Nicht unterstütztes Modell: {model_name}")


def build_model(
    config: PipelineConfig,
    num_classes: int,
) -> tuple[keras.Model, keras.Model]:
    """
    Baut das Gesamtmodell aus Augmentation, Preprocessing, Backbone und Head.
    Gibt sowohl das Gesamtmodell als auch das Backbone zurück.
    """
    inputs = keras.Input(
        shape=(config.img_size, config.img_size, 3),
        name="input_image",
    )

    if config.use_augmentation:
        x = build_augmentation_layer(config)(inputs)
    else:
        x = inputs

    preprocess_fn = get_preprocess_fn(config.model_name)
    x = preprocess_fn(x)

    base_model = build_backbone(config)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
    x = layers.Dropout(
        config.dropout,
        seed=config.seed,
        name="head_dropout",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"oct_{config.model_name.lower()}",
    )

    return model, base_model


def compile_model(
    model: keras.Model,
    learning_rate: float,
    num_classes: int,
) -> None:
    """Kompiliert das Modell mit Optimizer, Loss und Metriken."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=get_metrics(num_classes),
    )


def unfreeze_layers(base_model: keras.Model, unfreeze_last_n: int) -> None:
    """
    Aktiviert Fine-Tuning für die letzten N Layer des Backbones.
    BatchNorm-Layer bleiben eingefroren.
    """
    base_model.trainable = True
    split_idx = max(0, len(base_model.layers) - unfreeze_last_n)

    for i, layer in enumerate(base_model.layers):
        if i < split_idx or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True