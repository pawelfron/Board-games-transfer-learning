"""Self-play training loop and checkpoint writing."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from bgtl.checkpoint import save_checkpoint
from bgtl.device import require_cuda
from bgtl.games import make_game
from bgtl.model.res_conv_pool import ResConvConvLogitPoolModelV2
from bgtl.train.loss import policy_value_loss
from bgtl.train.replay import ReplayBuffer, Sample
from bgtl.train.self_play import self_play

CUSTOM_TRAINER_MESSAGE = (
    "trainer 'custom' is not implemented. Add a train entry that writes the shared "
    "checkpoint schema (model_state_dict, model_kwargs, config, game_id, board_size, "
    "state_channels, action_channels, step, games_played) and point this config at it. "
    "Eval and channel-map transfer stay shared via `bgtl eval` and `bgtl transfer`."
)


def train(config: dict) -> Path:
    trainer = config.get("trainer", "baseline")
    if trainer == "custom":
        raise SystemExit(CUSTOM_TRAINER_MESSAGE)
    if trainer != "baseline":
        raise SystemExit(f"Unknown trainer '{trainer}'.")

    device = require_cuda()
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_cfg = config["train"]
    model_cfg = config["model"]
    board_size = int(config["board_size"])
    win_length = int(config.get("win_length", 5))
    probe = make_game(config["game"], board_size, win_length)

    model = ResConvConvLogitPoolModelV2(
        c_state=len(probe.state_channels),
        c_action=len(probe.action_channels),
        nnsize=model_cfg["nnsize"],
        nb_nets=model_cfg["nb_nets"],
        nb_layers_per_net=model_cfg["nb_layers_per_net"],
        nnks=int(model_cfg.get("nnks", 3)),
        bn=bool(model_cfg.get("bn", True)),
        global_pooling=float(model_cfg.get("global_pooling", 0)),
        activation=str(model_cfg.get("activation", "relu")),
        batchnorm_momentum=float(model_cfg.get("batchnorm_momentum", 0.01)),
    ).to(device)
    if config.get("init_checkpoint"):
        from bgtl.checkpoint import load_checkpoint

        initial = load_checkpoint(config["init_checkpoint"])
        model.load_state_dict(initial["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    replay = ReplayBuffer(int(train_cfg["replay_capacity"]))
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    metrics_path = run_dir / "metrics.csv"
    csv_file = metrics_path.open("w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "games_played", "step", "policy_loss", "value_loss", "wall_time_s"],
    )
    csv_writer.writeheader()

    games_played = 0
    step = 0
    epoch = 0
    started = time.perf_counter()
    max_games = int(train_cfg["max_games"])
    max_minutes = train_cfg.get("max_minutes")
    latest = run_dir / "checkpoint_latest.pt"
    try:
        while games_played < max_games:
            if max_minutes is not None and (time.perf_counter() - started) >= float(max_minutes) * 60.0:
                print(f"Stopping: reached max_minutes={max_minutes}")
                break
            batch_games = min(int(train_cfg["games_per_iter"]), max_games - games_played)
            samples = self_play(
                model,
                board_size=board_size,
                win_length=win_length,
                n_games=batch_games,
                n_sims=int(train_cfg["n_sims"]),
                device=device,
                rng=rng,
                c_puct=float(train_cfg.get("c_puct", 1.25)),
                dirichlet_alpha=float(train_cfg.get("dirichlet_alpha", 0.3)),
                dirichlet_eps=float(train_cfg.get("dirichlet_eps", 0.25)),
            )
            replay.add(samples)
            games_played += batch_games
            policy_loss = ""
            value_loss = ""
            if len(replay) >= int(train_cfg["replay_warmup"]):
                model.train()
                policy_total = 0.0
                value_total = 0.0
                epoch_len = int(train_cfg["epoch_len"])
                for _ in range(epoch_len):
                    policy_err, value_err = _optimizer_step(
                        model,
                        optimizer,
                        replay.sample(int(train_cfg["batch_size"]), rng),
                        device,
                    )
                    policy_total += policy_err
                    value_total += value_err
                    step += 1
                policy_loss = policy_total / epoch_len
                value_loss = value_total / epoch_len
                writer.add_scalar("loss/policy", policy_loss, epoch)
                writer.add_scalar("loss/value", value_loss, epoch)
            wall = time.perf_counter() - started
            writer.add_scalar("games_played", games_played, epoch)
            writer.add_scalar("wall_time_s", wall, epoch)
            row = {
                "epoch": epoch,
                "games_played": games_played,
                "step": step,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "wall_time_s": f"{wall:.3f}",
            }
            csv_writer.writerow(row)
            csv_file.flush()
            print(
                f"epoch={epoch} games={games_played} step={step} "
                f"policy_loss={policy_loss} value_loss={value_loss} wall_s={wall:.1f}"
            )
            latest = _write_checkpoint(config, model, probe, run_dir, games_played, step, epoch)
            epoch += 1
    finally:
        csv_file.close()
        writer.close()
    return latest


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: list[Sample],
    device: torch.device,
) -> tuple[float, float]:
    states = torch.from_numpy(np.stack([sample.state for sample in batch])).to(device)
    policies = torch.from_numpy(np.stack([sample.policy for sample in batch])).to(device)
    values = torch.tensor([sample.value for sample in batch], dtype=torch.float32, device=device)
    masks = torch.from_numpy(np.stack([sample.legal_mask for sample in batch])).to(device)
    optimizer.zero_grad(set_to_none=True)
    pred_v, pred_logit = model(states)
    loss, value_err, policy_err = policy_value_loss(pred_v, pred_logit, values, policies, masks)
    loss.backward()
    optimizer.step()
    return float(policy_err.detach()), float(value_err.detach())


def _write_checkpoint(
    config: dict,
    model: ResConvConvLogitPoolModelV2,
    probe,
    run_dir: Path,
    games_played: int,
    step: int,
    epoch: int,
) -> Path:
    payload_extra = {
        "config": config,
        "game_id": probe.game_id,
        "board_size": probe.board_size,
        "win_length": probe.win_length,
        "state_channels": list(probe.state_channels),
        "action_channels": list(probe.action_channels),
        "step": step,
        "games_played": games_played,
        "epoch": epoch,
    }
    latest = run_dir / "checkpoint_latest.pt"
    save_checkpoint(latest, model, **payload_extra)
    save_checkpoint(run_dir / f"checkpoint_{games_played}.pt", model, **payload_extra)
    return latest
