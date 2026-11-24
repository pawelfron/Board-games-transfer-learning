# Board-games transfer learning

Training, transfer, and evaluation for fully convolutional policy-value networks, following Soemers et al., [Towards a General Transfer Approach for Policy-Value Networks](https://openreview.net/forum?id=vJcTm2v9Ku) (TMLR 2023). The local PDF is `../board_games_literature/1539_Towards_a_General_Transfe.pdf`.

The original scripts live in [Transfer-DNNs-Ludii-Polygames](https://github.com/DennisSoemers/Transfer-DNNs-Ludii-Polygames). Search and the network live in archived [Polygames](https://github.com/facebookarchive/Polygames), in particular [`pypolygames/model_zoo/res_conv_conv_logit_pool_model_v2.py`](https://github.com/facebookarchive/Polygames/blob/main/pypolygames/model_zoo/res_conv_conv_logit_pool_model_v2.py). That stack is Python 3.7 and CUDA 10.1, and it does not run on this machine's RTX 5060 Ti (`sm_120`). This project reimplements the algorithm: `ResConvConvLogitPoolModelV2`, PUCT self-play, the Polygames policy-value loss, and channel-index weight transfer.

## Setup

The project pins Python 3.12 and installs PyTorch from the CUDA 13.0 wheel index, which includes `sm_120` kernels.

```bash
uv sync
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list()); print(torch.nn.Conv2d(4, 8, 3, padding=1).cuda()(torch.randn(1, 4, 9, 9, device='cuda')).shape)"
```

Training and evaluation refuse to run without CUDA.

## First baseline

Gomoku, same rules at every board size. State channels are the side to move, the opponent, a constant to-move plane, and the previous move. The action head is one placement channel. Because the network is fully convolutional and the value head pools over the board, a 9×9 checkpoint and a 15×15 checkpoint have the same parameter tensors. Transfer between those sizes is an identity copy.

Appendix E uses `nnsize=2`, 10 residual groups, 6 layers per group, and batch norm. Polygames defaults used here are `global_pooling=0` (pooling remains in the value head), ReLU, batch-norm momentum `0.01`, and Adam with learning rate `1e-3`. Self-play uses 400 simulations. The loss is `1.5 * MSE(value) + cross-entropy(policy)`, with illegal logits shifted by `-400`.

Ludii's Gomoku tensor is wider than these four channels, so the paper reports 180,472 parameters. This representation has **36,186** parameters at the Appendix E depth. That count does not change between 9×9 and 15×15.

The paper trained for 20 hours on 8 GPUs. [`configs/gomoku9_baseline.yaml`](configs/gomoku9_baseline.yaml) and [`configs/gomoku15_baseline.yaml`](configs/gomoku15_baseline.yaml) keep the architecture and 400 simulations, and cap a run at 128 games or 10 hours on one GPU. Reported matches in the paper are 300 games at 800 simulations per side. The checked-in eval block uses the short protocol of 20 games and 50 simulations.

```bash
uv run bgtl train --config configs/gomoku9_baseline.yaml
uv run bgtl train --config configs/gomoku15_baseline.yaml
uv run bgtl transfer --config configs/gomoku9_to_15.yaml
uv run bgtl eval --config configs/gomoku9_to_15.yaml
```

Set `transfer.finetune: true` in [`configs/gomoku9_to_15.yaml`](configs/gomoku9_to_15.yaml) to keep training the copied 9×9 weights on 15×15. A channel index of `-1` marks a target channel with no source: zero-shot transfer fills it with zeros, fine-tuning leaves the fresh initialization. Pass `state_source_channels` and `move_source_channels` when the games do not share a channel layout.

## Smoke test

[`configs/smoke.yaml`](configs/smoke.yaml) trains a tiny net for 4 games on 9×9, copies it to 15×15, and plays 4 games.

```bash
uv run bgtl train --config configs/smoke.yaml
uv run bgtl transfer --config configs/smoke.yaml
uv run bgtl eval --config configs/smoke.yaml
```

Checkpoints and TensorBoard logs go under `runs/` and are not committed. Each checkpoint stores the weights, `model_kwargs`, the config, `game_id`, board size, and channel names.

## A second training method

[`configs/custom.yaml`](configs/custom.yaml) selects `trainer: custom`. `bgtl train` exits with an error until that trainer writes the same checkpoint schema. `bgtl eval` and `bgtl transfer` stay the comparison harness.
