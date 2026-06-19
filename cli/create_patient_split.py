from pathlib import Path
import csv
import random
from collections import Counter, defaultdict

DATA_ROOT = Path("data/OCT")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
OUTPUT_CSV = DATA_ROOT / "patient_split.csv"

VAL_SPLIT = 0.10
SEED = 42

CLASS_NAMES = {"CNV", "DME", "DRUSEN", "NORMAL"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract_patient_id(filename: str) -> str:
    """
    Extract the patient ID from the filename.

    Expected example:
    CNV-315649-12.jpeg -> patient_id = CNV-315649
    """
    stem = Path(filename).stem
    parts = stem.split("-")

    if len(parts) < 2:
        raise ValueError(f"Could not extract patient_id from filename: {filename}")

    return f"{parts[0]}-{parts[1]}"


def infer_label(path: Path) -> str:
    """
    Infer the class label from the directory structure.

    Expected example:
    data/OCT/train/CNV/image.jpeg -> label = CNV
    """
    for part in path.parts:
        if part.upper() in CLASS_NAMES:
            return part.upper()

    raise ValueError(f"Could not infer class label from path: {path}")


def collect_images(root: Path, split_name: str):
    """
    Collect all images below a given root directory and assign a fixed split name.
    """
    samples = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            label = infer_label(path)
            patient_id = extract_patient_id(path.name)

            samples.append(
                {
                    "filepath": str(path.as_posix()),
                    "filename": path.name,
                    "label": label,
                    "patient_id": patient_id,
                    "split": split_name,
                }
            )

    return samples


def split_train_patients_into_train_val(train_samples, val_split: float, seed: int):
    """
    Split patients from the training folder into train and validation sets.

    All images from the same patient stay in the same split.
    """
    patient_to_samples = defaultdict(list)
    for sample in train_samples:
        patient_to_samples[sample["patient_id"]].append(sample)

    patient_ids = list(patient_to_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n_patients = len(patient_ids)
    n_val = int(n_patients * val_split)

    val_patients = set(patient_ids[:n_val])
    train_patients = set(patient_ids[n_val:])

    split_samples = []
    for patient_id, samples in patient_to_samples.items():
        assigned_split = "val" if patient_id in val_patients else "train"

        for sample in samples:
            updated_sample = sample.copy()
            updated_sample["split"] = assigned_split
            split_samples.append(updated_sample)

    return split_samples, train_patients, val_patients


def build_patient_label_counts(rows):
    """
    Count unique patients per split and label.

    A patient is counted once per (split, label) combination, even if multiple
    images exist for that patient within the same class.
    """
    patient_label_count_per_split = defaultdict(Counter)
    seen = set()

    for row in rows:
        key = (row["split"], row["label"], row["patient_id"])
        if key not in seen:
            patient_label_count_per_split[row["split"]][row["label"]] += 1
            seen.add(key)

    return patient_label_count_per_split


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    train_folder_samples = collect_images(TRAIN_DIR, split_name="train")
    test_samples = collect_images(TEST_DIR, split_name="test")

    train_val_samples, train_patients, val_patients = split_train_patients_into_train_val(
        train_folder_samples,
        val_split=VAL_SPLIT,
        seed=SEED,
    )

    all_rows = train_val_samples + test_samples

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filepath", "filename", "label", "patient_id", "split"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    test_patients = {row["patient_id"] for row in test_samples}

    assert train_patients.isdisjoint(val_patients), "Leakage detected between train and val"
    assert train_patients.isdisjoint(test_patients), "Leakage detected between train and test"
    assert val_patients.isdisjoint(test_patients), "Leakage detected between val and test"

    image_count_per_split = Counter(row["split"] for row in all_rows)
    patient_count_per_split = {
        "train": len(train_patients),
        "val": len(val_patients),
        "test": len(test_patients),
    }

    label_count_per_split = defaultdict(Counter)
    for row in all_rows:
        label_count_per_split[row["split"]][row["label"]] += 1

    patient_label_count_per_split = build_patient_label_counts(all_rows)

    label_order = ["CNV", "DME", "DRUSEN", "NORMAL"]

    print(f"\nCSV saved to: {OUTPUT_CSV}")

    header = (
        f"{'Class':<10} "
        f"{'Train Patients':>14} {'Train Images':>14} "
        f"{'Val Patients':>12} {'Val Images':>12} "
        f"{'Test Patients':>13} {'Test Images':>12}"
    )

    print("\nSplit summary")
    print(header)
    print("-" * len(header))

    total_line = (
        f"{'TOTAL':<10} "
        f"{patient_count_per_split['train']:>14} {image_count_per_split['train']:>14} "
        f"{patient_count_per_split['val']:>12} {image_count_per_split['val']:>12} "
        f"{patient_count_per_split['test']:>13} {image_count_per_split['test']:>12}"
    )
    print(total_line)

    for label in label_order:
        line = (
            f"{label:<10} "
            f"{patient_label_count_per_split['train'][label]:>14} {label_count_per_split['train'][label]:>14} "
            f"{patient_label_count_per_split['val'][label]:>12} {label_count_per_split['val'][label]:>12} "
            f"{patient_label_count_per_split['test'][label]:>13} {label_count_per_split['test'][label]:>12}"
        )
        print(line)


if __name__ == "__main__":
    main()