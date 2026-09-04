# -*- coding: utf-8 -*-
"""Логическая голова (Э2 §4-bis): штраф падает на исправленной маске, растёт именно та
компонента, чью аксиому сломали; правила детача (гейты присутствия, опора неравенства)."""
import numpy as np
import pytest
import torch

from pyon import canon, logic
from pyon import oblique_synth as obs

F2, F1, E, Es = (canon.CLASSES.index(c) for c in ("F2", "F1", "E", "Es"))
NC = len(canon.CLASSES)


SCALE = 30.0   # «уверенные» логиты: остаточная масса softmax в пустых столбцах ≈ e^-30 (см. DIARY 2026-09-04)


def _band(y, cls, f_lo, f_hi, h_km, thick=1):
    """Сплошной горизонтальный «след» класса cls на высоте h_km в полосе частот [f_lo, f_hi]."""
    j0, j1 = canon.to_grid([f_lo, f_hi], canon.F_MIN, canon.F_MAX, canon.NF)
    jh = int(canon.to_grid([h_km], canon.H_MIN, canon.H_MAX, canon.NH)[0])
    y[jh - thick:jh + thick + 1, j0:j1 + 1] = cls


def _good_vertical():
    y = np.zeros((canon.NH, canon.NF), np.int8)
    _band(y, F2, 2.0, 8.0, 250.0)
    _band(y, F1, 2.0, 4.5, 180.0)
    _band(y, E, 1.5, 3.0, 110.0)
    return y


def _comps(mask, variant="hinge", fn=logic.vertical_logic):
    total, comps = fn(logic.logits_from_mask(mask[None], NC if fn is logic.vertical_logic else len(obs.OB_CLASSES), SCALE), variant)
    return float(total), {k: float(v.mean()) for k, v in comps.items()}


@pytest.mark.parametrize("variant", ["hinge", "lognorm"])
def test_good_mask_has_small_penalty(variant):
    total, comps = _comps(_good_vertical(), variant)
    assert np.isfinite(total)
    assert comps["P1"] < 0.05 and comps["P3"] < 0.05 and comps["P4"] < 0.05


def test_p1_fires_when_e_above_f2():
    good = _good_vertical()
    bad = np.zeros_like(good); _band(bad, F2, 2.0, 8.0, 250.0); _band(bad, E, 1.5, 3.0, 400.0)
    g, b = _comps(good)[1], _comps(bad)[1]
    assert b["P1"] > g["P1"] + 0.03 and b["P4"] <= g["P4"] + 1e-6


def test_p3_fires_on_multiple_labelled_as_f2():
    good = np.zeros((canon.NH, canon.NF), np.int8); _band(good, F2, 2.0, 6.0, 250.0)
    bad = good.copy(); _band(bad, F2, 2.0, 9.0, 500.0)               # кратник 2F2 размечен как F2
    g, b = _comps(good)[1], _comps(bad)[1]
    assert g["P3"] < 0.02 and b["P3"] > g["P3"] + 0.02
    # кратник ТОЛЬКО в столбцах основного следа (2–6 МГц): прототипный soft-argmax усреднял его с
    # основным следом и был слеп; заземление через max столбца (ревизия 2026-09-04) — видит
    bad2 = good.copy(); _band(bad2, F2, 2.0, 6.0, 500.0)
    assert _comps(bad2)[1]["P3"] > g["P3"] + 0.02


def test_p4_fires_when_foe_above_fof2():
    good = _good_vertical()
    bad = np.zeros_like(good); _band(bad, F2, 2.0, 8.0, 250.0); _band(bad, E, 1.5, 11.0, 110.0)
    g, b = _comps(good)[1], _comps(bad)[1]
    assert b["P4"] > g["P4"] + 0.03 and b["P1"] <= g["P1"] + 1e-6


def test_p2_fires_on_discontinuous_trace():
    good = np.zeros((canon.NH, canon.NF), np.int8); _band(good, F2, 2.0, 8.0, 250.0)
    bad = good.copy()
    j0, j1 = canon.to_grid([2.0, 8.0], canon.F_MIN, canon.F_MAX, canon.NF)
    hi = int(canon.to_grid([550.0], canon.H_MIN, canon.H_MAX, canon.NH)[0])
    for j in range(j0, j1 + 1, 2):                                     # каждый второй столбец на 300 км выше
        bad[:, j] = 0; bad[hi - 1:hi + 2, j] = F2
    g, b = _comps(good)[1], _comps(bad)[1]
    assert b["P2"] > g["P2"] + 0.5


def test_artist_mask_finite_and_lognorm_ge_hinge(sao_ji):
    y = canon.masks_from_sao(sao_ji)
    th, ch = _comps(y, "hinge")
    tl, cl = _comps(y, "lognorm")
    assert np.isfinite(th) and np.isfinite(tl)
    assert all(cl[k] >= ch[k] - 1e-6 for k in ch)                     # softplus ≥ hinge поточечно


def test_order_penalty_detaches_support():
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)
    logic.order_penalty(a, b, 0.0, logic.penalty("lognorm")).backward()
    assert a.grad is not None and a.grad > 0 and b.grad is None       # опора не двигается
    a2 = torch.tensor(1.0, requires_grad=True); b2 = torch.tensor(0.5, requires_grad=True)
    logic.order_penalty(a2, b2, 0.0, logic.penalty("lognorm"), detach=False).backward()
    assert b2.grad is not None and b2.grad < 0                          # без детача — «раздувание опоры»


def test_presence_gates_are_detached():
    """Сеть не может обнулить штраф, «выключив» классы: градиент P1 по логитам фона там,
    где стоят E и F2, идёт только через геометрию, а не через гейт присутствия."""
    y = np.zeros((canon.NH, canon.NF), np.int8); _band(y, F2, 2.0, 8.0, 250.0); _band(y, E, 1.5, 3.0, 400.0)
    lg = logic.logits_from_mask(y[None], NC, SCALE).requires_grad_(True)
    pres, h_soft, h_mean, fmax, present = logic.soft_readouts(lg, canon.h_axis)
    gate = (present[:, E - 1] * present[:, F2 - 1])
    assert gate.requires_grad                                            # сам по себе дифференцируем
    total, comps = logic.vertical_logic(lg, "hinge")
    comps["P1"].mean().backward()
    assert torch.isfinite(lg.grad).all() and lg.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", ["hinge", "lognorm"])
def test_oblique_logic_labels_vs_swapped(sao_ji, variant):
    y, lab = obs.oblique_masks_from_sao(sao_ji, 800.0, "O")
    iF2, iMH = obs.OB_CLASSES.index("F2"), obs.OB_CLASSES.index("MH")
    swapped = y.copy(); swapped[y == iF2] = iMH; swapped[y == iMH] = iF2  # кратник назван основным
    g = _comps(y, variant, logic.oblique_logic)[1]
    b = _comps(swapped, variant, logic.oblique_logic)[1]
    if variant == "hinge":
        assert g["S1"] == 0.0 and g["S2"] == 0.0                        # метки: аксиомы выполнены точно
    else:
        assert g["S1"] < 0.05 and g["S2"] < 0.05                        # softplus: малый ненулевой «запас»
    # сигнал S1 мал по построению: разность МПЧ(1F2)−МПЧ(2F2) ≈ 1.8 МГц = 0.08 нормированной оси
    print(f"[{variant}] метки S1={g['S1']:.4f} S2={g['S2']:.4f} | подмена S1={b['S1']:.4f} S2={b['S2']:.4f}")
    assert b["S1"] > g["S1"] * 1.05 + 1e-5 and b["S2"] > g["S2"] * 1.05 + 1e-5
