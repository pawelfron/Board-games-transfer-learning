"""Fully convolutional residual policy-value network.

Architecture follows Polygames ``ResConvConvLogitPoolModelV2`` (Cazenave et al.,
used by Soemers et al., TMLR 2023). Hidden width is ``int(nnsize * c_state)``.
With the paper default ``global_pooling=0``, global pooling is the value head's
adaptive max- and average-pool, which is what makes the weights independent of
board size. The optional per-block pool used by Polygames stays off unless
``global_pooling`` is set above 0.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class ResConvConvLogitPoolModelV2(nn.Module):
    def __init__(
        self,
        c_state: int,
        c_action: int,
        nnsize: float = 2,
        nb_nets: int = 10,
        nb_layers_per_net: int = 6,
        nnks: int = 3,
        bn: bool = True,
        global_pooling: float = 0,
        activation: str = "relu",
        batchnorm_momentum: float = 0.01,
    ) -> None:
        super().__init__()
        if c_state < 1 or c_action < 1:
            raise ValueError("c_state and c_action must be positive")
        self.c_state = int(c_state)
        self.c_action = int(c_action)
        self.nnsize = float(nnsize)
        self.nb_nets = int(nb_nets)
        self.nb_layers_per_net = int(nb_layers_per_net)
        self.nnks = int(nnks)
        self.bn = bool(bn)
        self.global_pooling = float(global_pooling)
        self.activation_name = activation
        self.batchnorm_momentum = float(batchnorm_momentum)
        self.hidden = int(self.nnsize * self.c_state)
        if self.hidden < 1:
            raise ValueError("nnsize * c_state rounds to an empty hidden width")
        padding = self.nnks // 2
        self.af = _activation(activation)

        stem: list[nn.Module] = [
            nn.Conv2d(
                self.c_state,
                self.hidden,
                self.nnks,
                stride=1,
                padding=padding,
                bias=not self.bn,
            )
        ]
        if self.bn:
            stem.append(
                nn.BatchNorm2d(
                    self.hidden,
                    momentum=self.batchnorm_momentum,
                    affine=True,
                )
            )
        self.mono = nn.Sequential(*stem)

        blocks: list[nn.ModuleList] = []
        for _block in range(self.nb_nets):
            layers: list[nn.Module] = []
            for layer_index in range(self.nb_layers_per_net):
                pool_extra = int(self.hidden * self.global_pooling) * 2 if layer_index == 0 else 0
                conv = nn.Conv2d(
                    self.hidden + pool_extra,
                    self.hidden,
                    self.nnks,
                    stride=1,
                    padding=padding,
                    bias=not self.bn,
                )
                if self.bn:
                    layers.append(
                        nn.Sequential(
                            conv,
                            nn.BatchNorm2d(
                                self.hidden,
                                momentum=self.batchnorm_momentum,
                                affine=True,
                            ),
                        )
                    )
                else:
                    layers.append(conv)
            blocks.append(nn.ModuleList(layers))
        self.resnets = nn.ModuleList(blocks)

        self.v = nn.Linear(2 * self.hidden, 2 * self.hidden)
        self.v2 = nn.Linear(2 * self.hidden, 1)
        self.pi_logit = nn.Conv2d(
            self.hidden,
            self.c_action,
            self.nnks,
            stride=1,
            padding=padding,
        )

    def model_kwargs(self) -> dict:
        return {
            "c_state": self.c_state,
            "c_action": self.c_action,
            "nnsize": self.nnsize,
            "nb_nets": self.nb_nets,
            "nb_layers_per_net": self.nb_layers_per_net,
            "nnks": self.nnks,
            "bn": self.bn,
            "global_pooling": self.global_pooling,
            "activation": self.activation_name,
            "batchnorm_momentum": self.batchnorm_momentum,
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        previous = self.mono(x)
        for resnet in self.resnets:
            hidden = previous
            if self.global_pooling > 0:
                width = int(hidden.size(1) * self.global_pooling)
                part = hidden.narrow(1, 0, width)
                hidden = torch.cat(
                    (
                        hidden,
                        F.adaptive_max_pool2d(part, 1).expand_as(part),
                        F.adaptive_avg_pool2d(part, 1).expand_as(part),
                    ),
                    dim=1,
                )
            hidden = self.af(hidden)
            for layer_index, layer in enumerate(resnet):
                if layer_index < self.nb_layers_per_net - 1:
                    hidden = self.af(layer(hidden))
                else:
                    hidden = layer(hidden) + previous
            previous = hidden
        hidden = self.af(previous)
        pooled = torch.cat(
            (F.adaptive_max_pool2d(hidden, 1), F.adaptive_avg_pool2d(hidden, 1)),
            dim=1,
        )
        value = torch.tanh(self.v2(self.af(self.v(pooled.flatten(1)))))
        policy_logits = self.pi_logit(hidden)
        return value, policy_logits


def _activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "celu":
        return F.celu
    raise ValueError(f"Unknown activation '{name}'")
