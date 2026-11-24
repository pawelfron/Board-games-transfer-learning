"""Channel remapping for zero-shot transfer."""

import torch

from bgtl.model.res_conv_pool import ResConvConvLogitPoolModelV2
from bgtl.transfer.channel_map import transfer_model


def _small(c_state: int, c_action: int) -> ResConvConvLogitPoolModelV2:
    return ResConvConvLogitPoolModelV2(
        c_state=c_state,
        c_action=c_action,
        nnsize=2,
        nb_nets=1,
        nb_layers_per_net=1,
        bn=True,
        global_pooling=0,
    )


def test_identity_copy_matches_and_plays_both_board_sizes():
    source = _small(4, 1)
    target = _small(4, 1)
    transfer_model(source, target, [0, 1, 2, 3], [0], zero_shot=True)
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])
    value9, logits9 = target(torch.zeros(1, 4, 9, 9))
    value15, logits15 = target(torch.zeros(1, 4, 15, 15))
    assert logits9.shape[-2:] == (9, 9)
    assert logits15.shape[-2:] == (15, 15)
    assert value9.shape == value15.shape == (1, 1)


def test_zero_shot_extra_channels_are_zero_and_known_channels_are_copied():
    source = _small(4, 1)
    target = _small(5, 2)
    with torch.no_grad():
        source.mono[0].weight.normal_()
        source.pi_logit.weight.normal_()
        source.pi_logit.bias.normal_()
    transfer_model(source, target, [0, 1, 2, 3, -1], [0, -1], zero_shot=True)
    stem_out = min(source.mono[0].weight.shape[0], target.mono[0].weight.shape[0])
    assert torch.count_nonzero(target.mono[0].weight[:, 4]) == 0
    assert torch.allclose(target.mono[0].weight[:stem_out, 0], source.mono[0].weight[:stem_out, 0])
    policy_in = min(source.pi_logit.weight.shape[1], target.pi_logit.weight.shape[1])
    assert torch.count_nonzero(target.pi_logit.weight[1]) == 0
    assert torch.allclose(target.pi_logit.weight[0, :policy_in], source.pi_logit.weight[0, :policy_in])
    assert torch.equal(target.pi_logit.bias[0], source.pi_logit.bias[0])
    assert target.pi_logit.bias[1].item() == 0
