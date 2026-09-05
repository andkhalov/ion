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


def test_profile_head():
    m = UNet(2, 5, profile=True)
    x = torch.rand(3, 2, 128, 128)
    logits, prof = m(x, profile=True)
    assert logits.shape == (3, 5, 128, 128) and prof.shape == (3, 2, 128)
    assert (prof[:, 0] >= 0).all()                     # fp ≥ 0 (softplus)
    assert m(x).shape == (3, 5, 128, 128)              # без профиля — как раньше


def test_profile_from_sao_and_scaler(sao_ji):
    import numpy as np
    from pyon import canon, scaler
    p = canon.profile_from_sao(sao_ji)
    ok = np.isfinite(p)
    assert ok.sum() > 20 and abs(canon.h_axis[np.nanargmax(p)] - sao_ji["scaled"]["hmF2"]) < 6
    assert abs(np.nanmax(p) - sao_ji["scaled"]["foF2"]) < 0.1
    y = canon.masks_from_sao(sao_ji)
    r = scaler.scale_vertical(y, p, f_b=canon.gyro_from_sao(sao_ji))
    assert abs(r["foF2"] - 8.2) <= 0.12 and abs(r["hmF2"] - 369.9) < 6
    assert abs(r["fxI"] - sao_ji["scaled"]["fxI"]) < 0.2               # 8.6 по fo²=fx(fx−fB), fB=0.8
    assert np.isfinite(r["MUF3000"]) and 0.97 < r["MUF3000"] / sao_ji["scaled"]["MUF3000F2"] < 1.03   # k = K_MUF = 1.11
    assert np.isnan(r["foE"]) and np.isnan(r["hmE"])
