"""Polygames policy-value loss: 1.5 * value MSE + policy cross-entropy."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def policy_value_loss(
    pred_v: torch.Tensor,
    pred_logit: torch.Tensor,
    target_v: torch.Tensor,
    target_pi: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(loss, value_loss, policy_loss)``, each averaged over the batch.

    Illegal logits are shifted by -400 before the softmax, matching
    ``pypolygames/model_zoo/loss.py``. Targets on illegal actions must be zero.
    """
    mask = legal_mask.to(dtype=pred_logit.dtype)
    logits = pred_logit * mask - 400.0 * (1.0 - mask)
    log_pi = F.log_softmax(logits.flatten(1), dim=1)
    policy_err = -(log_pi * target_pi.flatten(1)).sum(dim=1)
    value_err = F.mse_loss(pred_v.reshape(-1), target_v.reshape(-1), reduction="none")
    loss = 1.5 * value_err + policy_err
    return loss.mean(), value_err.mean(), policy_err.mean()
