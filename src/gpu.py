# src/gpu.py
from __future__ import annotations

import subprocess

import tensorflow as tf


def _get_gpu_memory_free() -> list[int]:
    """Liest den freien Speicher aller NVIDIA-GPUs per nvidia-smi aus."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return [int(x.strip()) for x in result.stdout.strip().splitlines() if x.strip()]


def select_gpu_index(requested_idx: int, num_gpus: int) -> int:
    """Bestimmt den zu verwendenden GPU-Index."""
    if num_gpus <= 0:
        raise ValueError("Es sind keine GPUs verfügbar.")

    if requested_idx == -1:
        try:
            free_memory = _get_gpu_memory_free()
            free_memory_valid = free_memory[:num_gpus]

            if free_memory_valid:
                return free_memory_valid.index(max(free_memory_valid))
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        return 0

    if 0 <= requested_idx < num_gpus:
        return requested_idx

    return 0


def configure_gpu(requested_idx: int = -1) -> str:
    """
    Konfiguriert TensorFlow für CPU oder genau eine GPU.

    Returns:
        String zur Beschreibung des genutzten Rechengeräts.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "CPU"

    target_idx = select_gpu_index(requested_idx, len(gpus))
    target_gpu = gpus[target_idx]

    try:
        tf.config.set_visible_devices(target_gpu, "GPU")
        tf.config.experimental.set_memory_growth(target_gpu, True)
        return target_gpu.name
    except RuntimeError as e:
        return f"GPU-Konfiguration fehlgeschlagen: {e}"
