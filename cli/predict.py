from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_OCT_CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict OCT classes for one image, multiple images, or folders.",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--images", type=str, nargs="*", default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--random", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--class_names", type=str, nargs="+", default=None)
    parser.add_argument(
        "--run_config",
        type=str,
        default=None,
        help="Optional path to a run_config.json with class_names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for prediction results.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_paths(args: argparse.Namespace) -> list[Path]:
    image_paths: list[Path] = []

    if args.images:
        image_paths.extend(Path(p) for p in args.images)

    if args.image_dir:
        image_dir = Path(args.image_dir)
        candidates = image_dir.rglob("*") if args.recursive else image_dir.glob("*")
        image_paths.extend(p for p in candidates if is_image_file(p))

    image_paths = [p.resolve() for p in image_paths if is_image_file(p)]
    if not image_paths:
        raise ValueError("No valid images found. Use --images, --image_dir, or both.")

    image_paths = sorted(set(image_paths))

    if args.random is not None:
        if args.random <= 0:
            raise ValueError("--random must be greater than 0.")
        random.seed(args.seed)
        image_paths = random.sample(image_paths, k=min(args.random, len(image_paths)))

    return image_paths


def candidate_run_config_paths(model_path: Path, explicit_path: str | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(model_path.parent / "model_card.json")
    candidates.append(model_path.parent / "best_summary.json")
    if len(model_path.parents) >= 2:
        candidates.append(model_path.parents[1] / "reports" / "metrics" / "run_config.json")
    return candidates


def load_class_names(
    model_path: Path,
    explicit_names: list[str] | None,
    run_config_path: str | None,
) -> list[str]:
    if explicit_names:
        return explicit_names

    for candidate in candidate_run_config_paths(model_path, run_config_path):
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        class_names = payload.get("class_names")
        if isinstance(class_names, list) and all(isinstance(x, str) for x in class_names):
            return class_names

    return DEFAULT_OCT_CLASS_NAMES


def load_image(image_path: Path, img_size: int) -> np.ndarray:
    from tensorflow import keras

    img = keras.utils.load_img(
        image_path,
        target_size=(img_size, img_size),
        color_mode="rgb",
    )
    return keras.utils.img_to_array(img)


def predict_images(
    model,
    image_paths: list[Path],
    img_size: int,
    class_names: list[str],
    batch_size: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    print()
    print("Predictions")
    print("=" * 100)

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = np.stack([load_image(path, img_size) for path in batch_paths], axis=0)
        predictions = model.predict(images, batch_size=batch_size, verbose=0)

        for image_path, probs in zip(batch_paths, predictions):
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx])
            row: dict[str, float | str] = {
                "image_path": str(image_path),
                "prediction": pred_class,
                "confidence": confidence,
            }

            probs_text_parts = []
            for class_name, prob in zip(class_names, probs):
                prob_float = float(prob)
                row[f"prob_{class_name}"] = prob_float
                probs_text_parts.append(f"{class_name}: {prob_float:.4f}")

            print(f"{image_path}")
            print(f"  -> Prediction   : {pred_class}")
            print(f"  -> Confidence   : {confidence:.4f}")
            print(f"  -> Probabilities: {' | '.join(probs_text_parts)}")
            print()

            rows.append(row)

    return rows


def write_predictions_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0.")

    from tensorflow import keras

    model_path = Path(args.model_path).resolve()
    model = keras.models.load_model(model_path, compile=False)
    img_size = int(model.input_shape[1])
    image_paths = collect_image_paths(args)
    class_names = load_class_names(model_path, args.class_names, args.run_config)

    output_size = int(model.output_shape[-1])
    if len(class_names) != output_size:
        raise ValueError(
            f"Number of class names ({len(class_names)}) does not match "
            f"model output size ({output_size})."
        )

    print(f"Loaded model : {model_path}")
    print(f"Images found : {len(image_paths)}")
    print(f"Classes      : {', '.join(class_names)}")

    rows = predict_images(model, image_paths, img_size, class_names, args.batch_size)
    if args.output is not None:
        write_predictions_csv(rows, args.output)
        print(f"CSV written  : {args.output}")


if __name__ == "__main__":
    main()
