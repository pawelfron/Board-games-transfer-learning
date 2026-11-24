"""Board-size invariance of ResConvConvLogitPoolModelV2."""

import torch

from bgtl.model.res_conv_pool import ResConvConvLogitPoolModelV2, count_parameters

PAPER = dict(
    c_state=4,
    c_action=1,
    nnsize=2,
    nb_nets=10,
    nb_layers_per_net=6,
    nnks=3,
    bn=True,
    global_pooling=0,
    activation="relu",
    batchnorm_momentum=0.01,
)


def test_paper_width_is_identical_at_9_and_15():
    model = ResConvConvLogitPoolModelV2(**PAPER)
    other = ResConvConvLogitPoolModelV2(**PAPER)
    count = count_parameters(model)
    assert count == count_parameters(other) == 36186
    for left, right in zip(model.state_dict().values(), other.state_dict().values()):
        assert left.shape == right.shape
    value9, logits9 = model(torch.zeros(2, 4, 9, 9))
    value15, logits15 = model(torch.zeros(2, 4, 15, 15))
    assert value9.shape == (2, 1)
    assert value15.shape == (2, 1)
    assert logits9.shape == (2, 1, 9, 9)
    assert logits15.shape == (2, 1, 15, 15)
    assert value9.abs().max() <= 1
    assert value15.abs().max() <= 1


def test_optional_block_pooling_keeps_spatial_size():
    model = ResConvConvLogitPoolModelV2(
        c_state=4,
        c_action=1,
        nnsize=2,
        nb_nets=1,
        nb_layers_per_net=2,
        global_pooling=0.5,
    )
    value, logits = model(torch.zeros(1, 4, 5, 5))
    assert value.shape == (1, 1)
    assert logits.shape == (1, 1, 5, 5)
