"""Checkpoint load and save. Weights are board-size invariant; metadata is not."""

from __future__ import annotations

from pathlib import Path

import torch

from bgtl.model.res_conv_pool import ResConvConvLogitPoolModelV2


def save_checkpoint(path: Path, model: ResConvConvLogitPoolModelV2, **metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model.model_kwargs(),
        **metadata,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def model_from_checkpoint(checkpoint: dict, device: torch.device | None = None) -> ResConvConvLogitPoolModelV2:
    model = ResConvConvLogitPoolModelV2(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model.to(device)
    return model
