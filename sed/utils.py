"""Utility helpers for the transformer-based SED MVP."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "sed") -> logging.Logger:
    """Return a simple stdout logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class Checkpoint:
    epoch: int
    model: Dict[str, torch.Tensor]
    optimizer: Dict[str, Any]
    extra: Dict[str, Any]


def save_checkpoint(path: Path, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, extra: Dict[str, Any]) -> None:
    """Persist model + optimizer state along with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "extra": extra,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: Optional[str | torch.device] = None) -> Checkpoint:
    """Load checkpoints produced by :func:`save_checkpoint`."""
    data = torch.load(path, map_location=map_location)
    return Checkpoint(
        epoch=data.get("epoch", 0),
        model=data["model"],
        optimizer=data.get("optimizer", {}),
        extra=data.get("extra", {}),
    )
