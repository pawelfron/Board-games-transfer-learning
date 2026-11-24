"""Copy stem input channels and policy output channels between games.

Index ``-1`` means the target channel has no source. In a zero-shot map those
parameters are set to zero. Otherwise they keep the target network's initialization.
Hidden layers are copied directly when shapes match, and by overlapping prefix
when a change in channel count also changes the hidden width (``nnsize * C``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from bgtl.checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint
from bgtl.games import make_game
from bgtl.model.res_conv_pool import ResConvConvLogitPoolModelV2


def transfer_model(
    source: ResConvConvLogitPoolModelV2,
    target: ResConvConvLogitPoolModelV2,
    state_source_channels: list[int] | None,
    move_source_channels: list[int] | None,
    zero_shot: bool,
) -> None:
    if state_source_channels is None:
        if source.c_state != target.c_state:
            raise ValueError("state_source_channels is required when the state channel count changes")
        state_source_channels = list(range(target.c_state))
    if move_source_channels is None:
        if source.c_action != target.c_action:
            raise ValueError("move_source_channels is required when the action channel count changes")
        move_source_channels = list(range(target.c_action))
    if len(state_source_channels) != target.c_state:
        raise ValueError("state_source_channels must have one entry per target state channel")
    if len(move_source_channels) != target.c_action:
        raise ValueError("move_source_channels must have one entry per target action channel")

    source_sd = source.state_dict()
    target_sd = {key: value.clone() for key, value in target.state_dict().items()}
    special = {"mono.0.weight", "mono.0.bias", "pi_logit.weight", "pi_logit.bias"}
    for key, src in source_sd.items():
        if key not in target_sd or key in special:
            continue
        if target_sd[key].shape == src.shape:
            target_sd[key] = src.detach().clone()
        else:
            _copy_overlapping(target_sd[key], src.detach())

    _remap_in_channels(
        target_sd,
        source_sd,
        weight_key="mono.0.weight",
        bias_key="mono.0.bias",
        source_indices=state_source_channels,
        zero_shot=zero_shot,
    )
    _remap_out_channels(
        target_sd,
        source_sd,
        weight_key="pi_logit.weight",
        bias_key="pi_logit.bias",
        source_indices=move_source_channels,
        zero_shot=zero_shot,
    )
    target.load_state_dict(target_sd)


def transfer(config: dict) -> Path:
    section = config["transfer"]
    source_ckpt = load_checkpoint(section["source_checkpoint"])
    board_size = int(section["target_board_size"])
    win_length = int(section.get("win_length", source_ckpt.get("win_length", 5)))
    game_name = section.get("game", source_ckpt.get("game_id", config.get("game", "gomoku")))
    game = make_game(game_name, board_size, win_length)
    kwargs = dict(source_ckpt["model_kwargs"])
    kwargs["c_state"] = len(game.state_channels)
    kwargs["c_action"] = len(game.action_channels)
    source = model_from_checkpoint(source_ckpt)
    target = ResConvConvLogitPoolModelV2(**kwargs)
    zero_shot = bool(section.get("zero_shot", True))
    transfer_model(
        source,
        target,
        _optional_indices(section.get("state_source_channels")),
        _optional_indices(section.get("move_source_channels")),
        zero_shot=zero_shot,
    )
    out = Path(section["out"])
    save_checkpoint(
        out,
        target,
        config=config,
        game_id=game.game_id,
        board_size=game.board_size,
        win_length=game.win_length,
        state_channels=list(game.state_channels),
        action_channels=list(game.action_channels),
        step=int(source_ckpt.get("step", 0)),
        games_played=int(source_ckpt.get("games_played", 0)),
        epoch=int(source_ckpt.get("epoch", 0)),
        transferred_from=str(section["source_checkpoint"]),
        zero_shot=zero_shot,
    )
    print(f"Wrote {out}")
    if section.get("finetune"):
        from bgtl.train.loop import train

        finetune_cfg = {
            "trainer": "baseline",
            "game": game.game_id,
            "board_size": game.board_size,
            "win_length": game.win_length,
            "seed": config.get("seed", 0),
            "run_dir": section.get("finetune_run_dir", str(out.parent / "finetune")),
            "model": {
                "nnsize": kwargs["nnsize"],
                "nb_nets": kwargs["nb_nets"],
                "nb_layers_per_net": kwargs["nb_layers_per_net"],
                "nnks": kwargs["nnks"],
                "bn": kwargs["bn"],
                "global_pooling": kwargs["global_pooling"],
                "activation": kwargs["activation"],
                "batchnorm_momentum": kwargs["batchnorm_momentum"],
            },
            "train": section["finetune_train"],
            "init_checkpoint": str(out),
        }
        return train(finetune_cfg)
    return out


def _optional_indices(value) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value]


def _remap_in_channels(
    target_sd: dict,
    source_sd: dict,
    weight_key: str,
    bias_key: str,
    source_indices: list[int],
    zero_shot: bool,
) -> None:
    if weight_key not in target_sd or weight_key not in source_sd:
        return
    dst = target_sd[weight_key]
    src = source_sd[weight_key]
    out = min(dst.shape[0], src.shape[0])
    k0 = min(dst.shape[2], src.shape[2])
    k1 = min(dst.shape[3], src.shape[3])
    for index, source_index in enumerate(source_indices):
        if source_index >= 0:
            dst[:out, index, :k0, :k1] = src[:out, source_index, :k0, :k1]
        elif zero_shot:
            dst[:, index].zero_()
    if bias_key in target_sd and bias_key in source_sd:
        _remap_vector(target_sd[bias_key], source_sd[bias_key], source_indices, zero_shot)


def _remap_out_channels(
    target_sd: dict,
    source_sd: dict,
    weight_key: str,
    bias_key: str,
    source_indices: list[int],
    zero_shot: bool,
) -> None:
    if weight_key not in target_sd or weight_key not in source_sd:
        return
    dst = target_sd[weight_key]
    src = source_sd[weight_key]
    inn = min(dst.shape[1], src.shape[1])
    k0 = min(dst.shape[2], src.shape[2])
    k1 = min(dst.shape[3], src.shape[3])
    for index, source_index in enumerate(source_indices):
        if source_index >= 0:
            dst[index, :inn, :k0, :k1] = src[source_index, :inn, :k0, :k1]
        elif zero_shot:
            dst[index].zero_()
    if bias_key in target_sd and bias_key in source_sd:
        _remap_vector(target_sd[bias_key], source_sd[bias_key], source_indices, zero_shot)


def _remap_vector(dst: torch.Tensor, src: torch.Tensor, source_indices: list[int], zero_shot: bool) -> None:
    for index, source_index in enumerate(source_indices):
        if source_index >= 0:
            dst[index] = src[source_index]
        elif zero_shot:
            dst[index] = 0


def _copy_overlapping(dst: torch.Tensor, src: torch.Tensor) -> None:
    dst_view = dst
    src_view = src
    while src_view.dim() < dst_view.dim():
        src_view = src_view.unsqueeze(0)
    while dst_view.dim() < src_view.dim():
        dst_view = dst_view.unsqueeze(0)
    slices = tuple(slice(0, min(dst_view.shape[dim], src_view.shape[dim])) for dim in range(dst_view.dim()))
    dst_view[slices].copy_(src_view[slices])
