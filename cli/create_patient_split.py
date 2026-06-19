# cli/create_patient_split_csv.py

from pathlib import Path
import random
import csv
from collections import defaultdict, Counter

DATA_ROOT = Path("data/OCT")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
OUTPUT_CSV = DATA_ROOT / "patient_split.csv"

VAL_SPLIT = 0.10
SEED = 42

CLASS_NAMES = {"CNV", "DME", "DRUSEN", "NORMAL"}


def extract_patient_id(filename: str) -> str:
    """
    Beispiel:
    CNV-315649-12.jpeg -> patient_id = CNV-315649
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 2:
        raise ValueError(f"Kann patient_id nicht aus Dateiname ableiten: {filename}")
    return f"{parts[0]}-{parts[1]}"


def infer_label(path: Path) -> str:
    for part in path.parts:
        if part.upper() in CLASS_NAMES:
            return part.upper()
    raise ValueError(f"Kein Klassenlabel im Pfad gefunden: {path}")


def collect_images_with_source_split(root: Path, fixed_split: str):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    samples = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            label = infer_label(path)
            patient_id = extract_patient_id(path.name)
            samples.append({
                "filepath": str(path.as_posix()),
                "filename": path.name,
                "label": label,
                "patient_id": patient_id,
                "split": fixed_split,
            })

    return samples


def split_train_patients_into_train_val(train_samples, val_split, seed):
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
        split_name = "val" if patient_id in val_patients else "train"
        for sample in samples:
            updated = sample.copy()
            updated["split"] = split_name
            split_samples.append(updated)

    return split_samples, train_patients, val_patients


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train-Ordner nicht gefunden: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test-Ordner nicht gefunden: {TEST_DIR}")

    raw_train_samples = collect_images_with_source_split(TRAIN_DIR, fixed_split="train")
    test_samples = collect_images_with_source_split(TEST_DIR, fixed_split="test")

    train_val_samples, train_patients, val_patients = split_train_patients_into_train_val(
        raw_train_samples,
        val_split=VAL_SPLIT,
        seed=SEED
    )

    all_rows = train_val_samples + test_samples

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filepath", "filename", "label", "patient_id", "split"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    test_patients = {row["patient_id"] for row in test_samples}

    assert train_patients.isdisjoint(val_patients), "Leakage zwischen train und val"
    assert train_patients.isdisjoint(test_patients), "Leakage zwischen train und test"
    assert val_patients.isdisjoint(test_patients), "Leakage zwischen val und test"

    image_count_per_split = Counter(row["split"] for row in all_rows)
    patient_count_per_split = {
        "train": len(train_patients),
        "val": len(val_patients),
        "test": len(test_patients),
    }

    label_count_per_split = defaultdict(Counter)
    for row in all_rows:
        label_count_per_split[row["split"]][row["label"]] += 1

    print(f"CSV gespeichert unter: {OUTPUT_CSV}")

    print("\nPatienten pro split:")
    for split, count in patient_count_per_split.items():
        print(f"  {split}: {count}")

    print("\nBilder pro split:")
    for split, count in image_count_per_split.items():
        print(f"  {split}: {count}")

    print("\nKlassenverteilung pro split:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {dict(label_count_per_split[split])}")


if __name__ == "__main__":
    main()