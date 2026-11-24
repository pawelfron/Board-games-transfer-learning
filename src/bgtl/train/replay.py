"""Fixed-capacity reservoir of self-play positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    state: np.ndarray
    policy: np.ndarray
    value: float
    legal_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._data: list[Sample] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._data)

    def add(self, samples: list[Sample]) -> None:
        for sample in samples:
            if len(self._data) < self.capacity:
                self._data.append(sample)
            else:
                self._data[self._pos] = sample
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Sample]:
        if not self._data:
            raise RuntimeError("cannot sample an empty replay buffer")
        indices = rng.integers(0, len(self._data), size=batch_size)
        return [self._data[int(index)] for index in indices]
