# inspect_dataset.py
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("data/OCT")  # adjust if needed
SPLITS = ["train", "test"]

# Pattern: CLASS-PATIENTID-Idx.ext
# Example:
#   CNV-232670-1.jpeg         -> patient_id = "232670"
#   NORMAL-ABC123_XY-5.jpeg   -> patient_id = "ABC123_XY"
PATTERN = re.compile(r"^[^-]+-(.+)-\d+$")


def parse_patient_id(filename: str) -> str | None:
    """
    Extract the patient ID from a filename of the form CLASS-PATIENTID-Idx.ext.

    Returns:
        patient_id (str) if the pattern matches, otherwise None.
    """
    stem = Path(filename).stem  # strip extension
    m = PATTERN.match(stem)
    if not m:
        return None
    return m.group(1)


def main() -> None:
    # Global map: patient_id -> set of splits it appears in
    patient_splits: dict[str, set[str]] = defaultdict(set)

    print(f"Root directory: {ROOT.resolve()}")

    for split in SPLITS:
        split_dir = ROOT / split
        if not split_dir.exists():
            print(f"\n[WARN] Split '{split}' does not exist ({split_dir})")
            continue

        print(f"\n=== {split.upper()} ===")

        # Per class: file counts and patient IDs
        class_file_counts: dict[str, int] = defaultdict(int)
        class_patient_ids: dict[str, set[str]] = defaultdict(set)
        unparsable = 0

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            for f in class_dir.iterdir():
                if not f.is_file():
                    continue
                class_file_counts[cls] += 1
                pid = parse_patient_id(f.name)
                if pid is None:
                    unparsable += 1
                    continue
                class_patient_ids[cls].add(pid)
                patient_splits[pid].add(split)

        total_files = sum(class_file_counts.values())
        total_patients = len({pid for s in class_patient_ids.values() for pid in s})

        print("Class       Files    Patients")
        print("--------------------------------")
        for cls in sorted(class_file_counts.keys()):
            n_files = class_file_counts[cls]
            n_patients = len(class_patient_ids[cls])
            print(f"{cls:<11}{n_files:8d}{n_patients:10d}")
        print("--------------------------------")
        print(f"{'TOTAL':<11}{total_files:8d}{total_patients:10d}")
        print(f"Unparsable filenames in {split}: {unparsable}")

    # Patients appearing in more than one split
    print("\n=== Patients in multiple splits (potential leakage) ===")
    leaks = {pid: splits for pid, splits in patient_splits.items() if len(splits) > 1}
    if not leaks:
        print("No patient IDs found in multiple splits.")
    else:
        print(f"{len(leaks)} patient IDs appear in multiple splits:")
        for pid, splits in sorted(leaks.items()):
            split_list = ", ".join(sorted(splits))
            print(f"  Patient {pid}: {split_list}")


if __name__ == "__main__":
    main()