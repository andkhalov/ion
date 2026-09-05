# -*- coding: utf-8 -*-
"""U-Net: дефолт = прототип (117 605 параметров), варианты архитектуры дают верные формы и градиент."""
import pytest
import torch

from pyon.models import UNet, n_params


def test_default_is_prototype():
    m = UNet(2, 5)
    assert n_params(m) == 117605
    assert m.depth == 3 and m.skip and not m.coords


@pytest.mark.parametrize("kw", [dict(depth=4), dict(base=32), dict(skip=False), dict(norm="group"),
                                dict(norm="none"), dict(dropout=0.2), dict(coords=True), dict(depth=2, base=8)])
def test_variants_forward_backward(kw):
    m = UNet(2, 5, **kw)
    x = torch.rand(2, 2, 128, 128, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 5, 128, 128)
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dropout_only_in_train_mode():
    m = UNet(2, 5, dropout=0.5)
    x = torch.rand(1, 2, 128, 128)
    m.eval()
    assert torch.allclose(m(x), m(x))                 # детерминирован в eval
    m.train()
    assert not torch.allclose(m(x), m(x))             # стохастичен в train
