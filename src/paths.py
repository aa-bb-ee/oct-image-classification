# src/paths.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config import PipelineConfig


@dataclass
class ExperimentPaths:
    run_id: str
    output_root: Path
    model_dir: Path
    log_dir: Path
    fig_dir: Path
    met_dir: Path
    best_model_path: Path

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "ExperimentPaths":
        project_root = Path(__file__).resolve().parents[1]
        run_id = generate_run_id(config)

        output_root = project_root / config.output_root_name / run_id
        model_dir = output_root / "models"
        log_dir = output_root / "runs"
        fig_dir = output_root / "reports" / "figures"
        met_dir = output_root / "reports" / "metrics"
        best_model_path = model_dir / "best_model.keras"

        return cls(
            run_id=run_id,
            output_root=output_root,
            model_dir=model_dir,
            log_dir=log_dir,
            fig_dir=fig_dir,
            met_dir=met_dir,
            best_model_path=best_model_path,
        )

    def create_directories(self) -> None:
        for directory in (self.model_dir, self.log_dir, self.fig_dir, self.met_dir):
            directory.mkdir(parents=True, exist_ok=True)


def generate_run_id(config: PipelineConfig) -> str:
    timestamp = datetime.now().strftime(config.timestamp_format)
    return f"{timestamp}_{config.run_name}"
