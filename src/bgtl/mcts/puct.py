"""Batched PUCT. One forward pass evaluates every leaf in the batch."""

from __future__ import annotations

import math

import numpy as np
import torch

from bgtl.games.gomoku import Gomoku


class _Edge:
    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.n = 0
        self.w = 0.0
        self.child = _Node()


class _Node:
    def __init__(self) -> None:
        self.expanded = False
        self.edges: dict[int, _Edge] = {}

    def expand(
        self,
        legal: np.ndarray,
        prior: np.ndarray,
        rng: np.random.Generator,
        noise: bool,
        dirichlet_alpha: float,
        dirichlet_eps: float,
    ) -> None:
        self.expanded = True
        priors = prior.astype(np.float64, copy=True)
        if legal.size == 0:
            return
        if noise:
            draw = rng.dirichlet(np.full(legal.size, dirichlet_alpha))
            priors[legal] = (1.0 - dirichlet_eps) * priors[legal] + dirichlet_eps * draw
        total = float(priors[legal].sum())
        if total <= 0.0:
            priors[legal] = 1.0 / legal.size
        else:
            priors[legal] /= total
        for action in legal.tolist():
            self.edges[int(action)] = _Edge(float(priors[int(action)]))

    def select(self, c_puct: float) -> int:
        total_n = sum(edge.n for edge in self.edges.values())
        # +1 keeps the first visit ordered by the prior instead of a tie at zero.
        sqrt_total = math.sqrt(total_n + 1.0)
        best_action = -1
        best_score = -1e30
        for action, edge in self.edges.items():
            mean = edge.w / edge.n if edge.n else 0.0
            bonus = c_puct * edge.prior * sqrt_total / (1 + edge.n)
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_action = action
        return best_action


def run_mcts(
    games: list[Gomoku],
    model: torch.nn.Module,
    n_sims: int,
    device: torch.device,
    rng: np.random.Generator,
    c_puct: float = 1.25,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    add_noise: bool = False,
) -> list[np.ndarray]:
    """Return a visit-count policy, shaped like each game's legal mask, for every root."""
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    if not games:
        return []
    roots = [_Node() for _ in games]
    _values, policies = _evaluate(games, model, device)
    del _values
    for root, game, policy in zip(roots, games, policies):
        root.expand(
            _legal_indices(game),
            policy.reshape(-1),
            rng,
            noise=add_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
        )

    for _ in range(n_sims):
        leaves: list[_Node] = []
        leaf_games: list[Gomoku] = []
        paths: list[list[tuple[_Node, int]]] = []
        for root, game in zip(roots, games):
            node = root
            cursor = game.clone()
            path: list[tuple[_Node, int]] = []
            while node.expanded and node.edges and not cursor.terminal():
                action = node.select(c_puct)
                path.append((node, action))
                cursor.apply(action)
                node = node.edges[action].child
            leaves.append(node)
            leaf_games.append(cursor)
            paths.append(path)

        pending = [index for index, game in enumerate(leaf_games) if not game.terminal()]
        eval_values: dict[int, float] = {}
        eval_policies: dict[int, np.ndarray] = {}
        if pending:
            batch_values, batch_policies = _evaluate([leaf_games[index] for index in pending], model, device)
            for index, value, policy in zip(pending, batch_values, batch_policies):
                eval_values[index] = float(value)
                eval_policies[index] = policy

        for index, (node, game, path) in enumerate(zip(leaves, leaf_games, paths)):
            if game.terminal():
                value = game.outcome()
            else:
                value = eval_values[index]
                if not node.expanded:
                    node.expand(
                        _legal_indices(game),
                        eval_policies[index].reshape(-1),
                        rng,
                        noise=False,
                        dirichlet_alpha=dirichlet_alpha,
                        dirichlet_eps=dirichlet_eps,
                    )
            for parent, action in reversed(path):
                value = -value
                edge = parent.edges[action]
                edge.n += 1
                edge.w += value

    results: list[np.ndarray] = []
    for root, game in zip(roots, games):
        counts = np.zeros(game.legal_mask().shape, dtype=np.float32).reshape(-1)
        for action, edge in root.edges.items():
            counts[action] = edge.n
        total = float(counts.sum())
        if total <= 0.0:
            legal = _legal_indices(game)
            if legal.size:
                counts[legal] = 1.0 / legal.size
        else:
            counts /= total
        results.append(counts.reshape(game.legal_mask().shape))
    return results


def _legal_indices(game: Gomoku) -> np.ndarray:
    return np.flatnonzero(game.legal_mask().reshape(-1))


def _evaluate(
    games: list[Gomoku],
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    states = np.stack([game.state_tensor() for game in games]).astype(np.float32, copy=False)
    masks = np.stack([game.legal_mask() for game in games]).astype(np.float32, copy=False)
    state_t = torch.from_numpy(states).to(device)
    mask_t = torch.from_numpy(masks).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        values, logits = model(state_t)
        # Same illegal-logit shift as the Polygames policy loss.
        masked = logits * mask_t - 400.0 * (1.0 - mask_t)
        flat = torch.softmax(masked.flatten(1), dim=1)
        policies = flat.reshape(masked.shape)
    model.train(was_training)
    policy_np = policies.detach().cpu().numpy().astype(np.float32)
    return values.detach().cpu().numpy().reshape(-1), [policy_np[index] for index in range(len(games))]
