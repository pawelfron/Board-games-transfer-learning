"""CUDA device selection. Training and evaluation do not fall back to CPU."""

from __future__ import annotations

import torch


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Refusing to fall back to CPU.")
    device = torch.device("cuda:0")
    arch = torch.cuda.get_arch_list()
    if not any("120" in entry for entry in arch):
        raise SystemExit(f"This GPU needs an sm_120 PyTorch build. torch.cuda.get_arch_list()={arch}")
    name = torch.cuda.get_device_name(device)
    print(f"Using {name} on {device} arch={arch}")
    return device
