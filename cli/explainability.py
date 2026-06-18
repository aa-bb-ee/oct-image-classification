# cli/explainability.py
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.helpers import ensure_dir, print_kv, print_section


# ---------- CLI ---------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualizations for OCT classification models.",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--test_subdir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_per_class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


# ---------- CONFIG ---------- #

def build_config(args: argparse.Namespace, img_size: int) -> PipelineConfig:
    config_fields = {field.name for field in fields(PipelineConfig)}
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in config_fields and value is not None
    }
    overrides["img_size"] = img_size
    return PipelineConfig(**overrides)


# ---------- DATA ---------- #

def collect_images(
    config: PipelineConfig,
    num_per_class: int,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if not config.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {config.test_dir}")

    class_dirs = sorted(
        [p for p in config.test_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    class_names = [p.name for p in class_dirs]

    out_imgs: list[np.ndarray] = []
    out_labels: list[int] = []

    for class_idx, class_dir in enumerate(class_dirs):
        image_paths = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            ]
        )

        if not image_paths:
            continue

        chosen = random.sample(
            image_paths,
            k=min(num_per_class, len(image_paths)),
        )

        for image_path in chosen:
            img_bytes = tf.io.read_file(str(image_path))
            img = tf.image.decode_image(
                img_bytes,
                channels=3,
                expand_animations=False,
            )
            img = tf.image.resize(img, [config.img_size, config.img_size])
            img = tf.cast(img, tf.uint8).numpy()

            out_imgs.append(img)
            out_labels.append(class_idx)

    return out_imgs, out_labels, class_names


# ---------- GRAD-CAM ---------- #

def find_gradcam_layer(model: keras.Model) -> keras.layers.Layer:
    """Sucht erst 'gradcam_features', sonst den letzten 4D-Layer."""
    try:
        return model.get_layer("gradcam_features")
    except ValueError:
        pass

    candidates: list[keras.layers.Layer] = []
    for layer in model.layers:
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None and hasattr(layer, "output"):
            output_shape = getattr(layer.output, "shape", None)
        if output_shape is not None and len(output_shape) == 4:
            candidates.append(layer)

    if not candidates:
        raise ValueError(
            "No 4D feature-map layer found. Train a compatible model first.",
        )

    return candidates[-1]


def _call_layer(layer: keras.layers.Layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def build_grad_model(
    model: keras.Model,
    gradcam_layer: keras.layers.Layer,
) -> keras.Model:
    """Hilfsmodell mit Outputs: (feature maps, predictions)."""
    x = model.inputs[0]
    gradcam_output = None

    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue

        x = _call_layer(layer, x)
        if layer.name == gradcam_layer.name:
            gradcam_output = x

    if gradcam_output is None:
        raise ValueError(
            f"Could not reconnect Grad-CAM layer '{gradcam_layer.name}' "
            "inside the loaded model graph.",
        )

    return keras.Model(
        inputs=model.inputs,
        outputs=[gradcam_output, x],
        name=f"{model.name}_gradcam",
    )


def make_gradcam_heatmap(
    model: keras.Model,
    img_array: np.ndarray,
    class_idx: int | None = None,
) -> np.ndarray:
    """
    Berechnet eine Grad-CAM Heatmap.

    img_array: shape (1, H, W, 3), float32/uint8.
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    gradcam_layer = find_gradcam_layer(model)
    grad_model = build_grad_model(model, gradcam_layer)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise ValueError("Could not compute gradients for Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    conv_out_0 = conv_out[0]
    pooled_grads_0 = pooled_grads[0]

    heatmap = tf.reduce_sum(conv_out_0 * pooled_grads_0, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()


def overlay(img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    heatmap_resized = tf.image.resize(
        heatmap[..., None],
        (h, w),
    ).numpy().squeeze()
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    colored = (plt.get_cmap("jet")(heatmap_resized)[..., :3] * 255).astype("uint8")
    blended = img.astype("float32") * 0.6 + colored.astype("float32") * 0.4
    return np.clip(blended, 0, 255).astype("uint8")


# ---------- OUTPUT PATH ---------- #

def default_output_dir(model_path: Path) -> Path:
    if len(model_path.parents) >= 2:
        return model_path.parents[1] / "reports" / "figures" / "grad_cam"
    return Path("grad_cam")


# ---------- MAIN ---------- #

def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path).resolve()

    print_section("Loading Model")
    model = keras.models.load_model(model_path, compile=False)
    print_kv("Model", model.name)
    print_kv("Input Shape", model.input_shape)
    print_kv("Grad-CAM Layer", find_gradcam_layer(model).name)

    img_size = int(model.input_shape[1])
    config = build_config(args, img_size=img_size)
    random.seed(args.seed if args.seed is not None else config.seed)

    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(model_path)
    ensure_dir(out_dir)

    print_section("Loading Test Images")
    imgs, labels, class_names = collect_images(config, args.num_per_class)
    print_kv("Classes", ", ".join(class_names))
    print_kv("Images", len(imgs))

    print_section("Generating Grad-CAM")
    for idx, (img, label) in enumerate(zip(imgs, labels)):
        class_idx = int(label)
        class_name = class_names[class_idx]

        x = np.expand_dims(img.astype("float32"), axis=0)
        heatmap = make_gradcam_heatmap(model, x, class_idx)
        overlay_img = overlay(img, heatmap)

        out_path = out_dir / f"gradcam_{idx:03d}_{class_name}.png"

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"{class_name} (true)")
        axes[0].axis("off")

        axes[1].imshow(overlay_img)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print_kv("Saved", out_path)


if __name__ == "__main__":
    main()