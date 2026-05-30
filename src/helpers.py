# src/helpers.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def print_section(title: str, width: int = 78) -> None:
    """Gibt eine optisch hervorgehobene Überschrift im Terminal aus."""
    line = "=" * width
    print(f"\n{line}\n{title}\n{line}")


def print_kv(label: str, value: Any, label_width: int = 32) -> None:
    """Gibt ein sauber formatiertes Schlüssel-Wert-Paar im Terminal aus."""
    print(f"{label:<{label_width}}: {value}")


def ensure_dir(path: Path) -> Path:
    """Stellt sicher, dass ein Verzeichnis existiert und gibt den Pfad zurück."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict[str, Any], indent: int = 4) -> None:
    """Speichert ein Dictionary sauber formatiert als JSON-Datei ab."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    """Lädt eine JSON-Datei und gibt sie als Dictionary zurück."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def merge_histories(h1: Any, h2: Any | None = None) -> dict[str, list[float]]:
    """Verbindet die Trainings-Histories von Stage 1 und Stage 2 nahtlos."""

    def _to_history_dict(history_obj: Any) -> dict[str, list[float]]:
        if hasattr(history_obj, "history"):
            history_obj = history_obj.history

        if not isinstance(history_obj, dict):
            raise TypeError("History muss ein Keras-History-Objekt oder ein dict sein.")

        return {k: list(v) for k, v in history_obj.items()}

    merged = _to_history_dict(h1)

    if h2 is not None:
        h2_dict = _to_history_dict(h2)
        for k, v in h2_dict.items():
            if k in merged:
                merged[k].extend(v)
            else:
                merged[k] = list(v)

    return merged
