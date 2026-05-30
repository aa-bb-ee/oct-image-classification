# src/config.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    # ------------------ Projekt / Daten ------------------
    data_dir: Path = Path("data/OCT")
    train_subdir: str = "train"
    test_subdir: str = "test"

    img_size: int = 299
    batch_size: int = 32
    val_split: float = 0.1
    seed: int = 42

    # ------------------ Modell ------------------
    model_name: str = "inceptionv3"

    # ------------------ Training ------------------
    epochs: int = 10
    fine_tune_epochs: int = 10

    learning_rate: float = 1e-4
    fine_tune_lr: float = 1e-5

    dropout: float = 0.3
    unfreeze_last_n: int = 50

    # ------------------ Laufsteuerung ------------------
    run_name: str = "oct_exp"
    gpu_index: int = -1

    train_take: int = -1
    val_take: int = -1
    test_take: int = -1

    cache: bool = False
    mixed_precision: bool = False
    fine_tune: bool = False
    use_class_weights: bool = False
    use_augmentation: bool = True

    # ------------------ Projektmetadaten ------------------
    project_name: str = "oct_project"
    output_root_name: str = "experiment_outputs"
    timestamp_format: str = "%Y%m%d_%H%M%S"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)

        if not self.run_name.strip():
            raise ValueError("run_name darf nicht leer sein.")

        if not self.model_name.strip():
            raise ValueError("model_name darf nicht leer sein.")

        if not (0.0 < self.val_split < 1.0):
            raise ValueError("val_split muss zwischen 0 und 1 liegen.")

        if self.img_size <= 0:
            raise ValueError("img_size muss > 0 sein.")

        if self.batch_size <= 0:
            raise ValueError("batch_size muss > 0 sein.")

        if self.epochs < 0:
            raise ValueError("epochs darf nicht negativ sein.")

        if self.fine_tune_epochs < 0:
            raise ValueError("fine_tune_epochs darf nicht negativ sein.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate muss > 0 sein.")

        if self.fine_tune_lr <= 0:
            raise ValueError("fine_tune_lr muss > 0 sein.")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout muss im Intervall [0, 1) liegen.")

        if self.unfreeze_last_n < 0:
            raise ValueError("unfreeze_last_n darf nicht negativ sein.")

        for attr_name in ("train_take", "val_take", "test_take"):
            value = getattr(self, attr_name)
            if value == 0 or value < -1:
                raise ValueError(f"{attr_name} muss -1 oder > 0 sein.")

    @property
    def train_dir(self) -> Path:
        return self.data_dir / self.train_subdir

    @property
    def test_dir(self) -> Path:
        return self.data_dir / self.test_subdir

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["train_dir"] = str(self.train_dir)
        payload["test_dir"] = str(self.test_dir)
        return payload
