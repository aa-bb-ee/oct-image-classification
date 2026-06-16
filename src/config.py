# src/config.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    # ------------------ Project / Data ------------------
    data_dir: Path = Path("data/OCT")
    train_subdir: str = "train"
    test_subdir: str = "test"

    img_size: int = 299
    batch_size: int = 32
    val_split: float = 0.1
    seed: int = 42

    # ------------------ Model ------------------
    model_name: str = "inceptionv3"

    # ------------------ Training ------------------
    epochs: int = 10
    fine_tune_epochs: int = 10

    learning_rate: float = 1e-4
    fine_tune_lr: float = 1e-5

    dropout: float = 0.3
    unfreeze_last_n: int = 50

    # ------------------ Run control ------------------
    run_name: str | None = None
    gpu_index: int = -1

    train_take: int = -1
    val_take: int = -1
    test_take: int = -1

    cache: bool = False
    mixed_precision: bool = False
    fine_tune: bool = False
    use_class_weights: bool = False
    use_augmentation: bool = True

    # ------------------ Project metadata ------------------
    project_name: str = "oct_project"
    output_root_name: str = "experiment_outputs"
    timestamp_format: str = "%Y%m%d_%H%M%S"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)

        if self.run_name is not None and not self.run_name.strip():
            raise ValueError("run_name must not be empty.")

        if self.run_name is None:
            self.run_name = self.build_run_name()

        if not self.model_name.strip():
            raise ValueError("model_name must not be empty.")

        if not (0.0 < self.val_split < 1.0):
            raise ValueError("val_split must be between 0 and 1.")

        if self.img_size <= 0:
            raise ValueError("img_size must be > 0.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        if self.epochs < 0:
            raise ValueError("epochs must not be negative.")

        if self.fine_tune_epochs < 0:
            raise ValueError("fine_tune_epochs must not be negative.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")

        if self.fine_tune_lr <= 0:
            raise ValueError("fine_tune_lr must be > 0.")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in the interval [0, 1).")

        if self.unfreeze_last_n < 0:
            raise ValueError("unfreeze_last_n must not be negative.")

        for attr_name in ("train_take", "val_take", "test_take"):
            value = getattr(self, attr_name)
            if value == 0 or value < -1:
                raise ValueError(f"{attr_name} must be -1 or > 0.")

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

    def build_run_name(self) -> str:
        """
        Automatically generate a descriptive experiment name.
        """
        model_map = {
            "inceptionv3": "incv3",
            "resnet50": "res50",
            "efficientnetb0": "effb0",
            "efficientnetb3": "effb3",
            "cnn_scratch": "cnn",
            "inceptionv4": "incv4",
        }

        model_part = model_map.get(
            self.model_name.lower(),
            self.model_name.lower(),
        )

        if self.model_name.lower() == "cnn_scratch":
            train_part = "scratch"
        elif self.fine_tune:
            train_part = f"ft{self.unfreeze_last_n}"
        else:
            train_part = "frozen"

        cw_part = "cw" if self.use_class_weights else "nocw"
        aug_part = "aug" if self.use_augmentation else "noaug"

        return "_".join(
            [
                model_part,
                train_part,
                f"img{self.img_size}",
                f"bs{self.batch_size}",
                f"e{self.epochs}",
                f"lr{self.learning_rate:.0e}",
                f"do{int(self.dropout * 100)}",
                cw_part,
                aug_part,
            ]
        )