# inspect_dataset.py
from __future__ import annotations

from collections import defaultdict
from pathlib import Path


ROOT = Path("data/OCT")  # ggf. anpassen
SPLITS = ["train", "test"]  # vorhandene Splits werden automatisch erkannt


def parse_patient_id(filename: str) -> str | None:
    """
    Erwartetes Format: KLASSENNAME-PATIENTID-Nr.jpeg
    Beispiel: CNV-232670-1.jpeg -> patient_id = "232670"
    """
    stem = Path(filename).stem  # ohne .jpeg
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    return parts[1]


def main() -> None:
    # global: patient_id -> set(splits)
    patient_splits: dict[str, set[str]] = defaultdict(set)

    print(f"Root: {ROOT.resolve()}")

    for split in SPLITS:
        split_dir = ROOT / split
        if not split_dir.exists():
            print(f"\n[WARN] Split '{split}' existiert nicht ({split_dir})")
            continue

        print(f"\n=== {split.upper()} ===")

        # pro Klasse: Dateianzahl und Patient-IDs
        class_file_counts: dict[str, int] = defaultdict(int)
        class_patient_ids: dict[str, set[str]] = defaultdict(set)

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            for f in class_dir.iterdir():
                if not f.is_file():
                    continue
                class_file_counts[cls] += 1
                pid = parse_patient_id(f.name)
                if pid is not None:
                    class_patient_ids[cls].add(pid)
                    patient_splits[pid].add(split)

        total_files = sum(class_file_counts.values())
        total_patients = len({pid for s in class_patient_ids.values() for pid in s})

        print("Klasse       Dateien   Patienten")
        print("--------------------------------")
        for cls in sorted(class_file_counts.keys()):
            n_files = class_file_counts[cls]
            n_patients = len(class_patient_ids[cls])
            print(f"{cls:<12}{n_files:8d}{n_patients:11d}")
        print("--------------------------------")
        print(f"{'TOTAL':<12}{total_files:8d}{total_patients:11d}")

    # Patienten, die in mehreren Splits vorkommen
    print("\n=== Patienten in mehreren Splits (potenzielles Leakage) ===")
    leaks = {pid: splits for pid, splits in patient_splits.items() if len(splits) > 1}
    if not leaks:
        print("Keine Patient-IDs in mehreren Splits gefunden.")
    else:
        print(f"{len(leaks)} Patient-IDs kommen in mehreren Splits vor:")
        for pid, splits in sorted(leaks.items()):
            split_list = ", ".join(sorted(splits))
            print(f"  Patient {pid}: {split_list}")


if __name__ == "__main__":
    main()