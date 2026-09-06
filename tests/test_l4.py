# -*- coding: utf-8 -*-
"""L4-интерпретация и синтетика: измеритель scaler, рендерер (E4), НЗ-ридауты (E5)."""
import numpy as np
import pytest
import torch

from pyon import canon, renderer, scaler
from pyon import oblique_synth as obs


def test_trace_from_mask_and_muf_table(sao_ji):
    y = canon.masks_from_sao(sao_ji)
    f, h = scaler.trace_from_mask(y, canon.CLASSES.index("F2"))
    assert len(f) > 30 and np.all(np.diff(f) > 0) and h.min() >= 200 and h.max() <= 600
    r = scaler.scale_vertical(y, None, f_b=0.8)
    assert all(np.isfinite(r[f"MUF{D}"]) for D in scaler.MUF_DISTANCES)
    assert r["MUF100"] < r["MUF800"] < r["MUF3000"]                       # МПЧ растёт с дальностью
    assert abs(r["M3000F2"] - r["MUF3000"] / r["foF2"]) < 1e-9
    assert np.isnan(r["hmF2"])                                             # без профиля hm* нет
    assert scaler.fx_from_fo(8.2, 0.8) == pytest.approx(8.61, abs=0.02)


def test_profile_peaks_day_like():
    fp = np.full(canon.NH, np.nan, np.float32)
    h = canon.h_axis
    fp[(h >= 95) & (h <= 250)] = 3.0 + 0.02 * (h[(h >= 95) & (h <= 250)] - 95)      # монотонный рост
    fp[(h >= 100) & (h <= 110)] += 0.5                                                # локальный максимум E
    pk = scaler.profile_peaks(fp)
    assert abs(pk["hmF2"] - 250) < 6 and len(pk["peaks"]) >= 1 and 95 <= pk["peaks"][0][0] <= 115


def test_renderer_shapes_and_sampling():
    y = torch.zeros(2, 128, 128, dtype=torch.long); y[:, 40:43, 20:60] = 1
    x = torch.rand(2, 2, 128, 128) * (torch.rand(2, 2, 128, 128) > 0.9)
    net = renderer.Renderer(base=8, depth=2)                                   # прототипный режим (старые чекпойнты)
    inp = net.make_input(y)
    assert inp.shape == (2, renderer.N_CLASSES + 3, 128, 128) and inp[:, -1].min() >= 0 and inp[:, -1].max() <= 1
    out = net(y); assert out.shape == (2, 4, 128, 128)
    loss = renderer.render_loss(out, x); assert torch.isfinite(loss) and loss > 0
    s = net.sample(y); assert s.shape == (2, 2, 128, 128) and s.min() >= 0 and s.max() <= 1
    net = renderer.Renderer(base=8, depth=2, hetero=True, col_noise=True)      # E4: (μ, log σ) + постолбцовый шум
    inp = net.make_input(y); assert inp.shape == (2, renderer.N_CLASSES + 4, 128, 128)
    assert torch.equal(inp[:, -1, 0], inp[:, -1, 77])                           # z_col одинаков по высоте
    out = net(y); assert out.shape == (2, 6, 128, 128)
    assert torch.isfinite(renderer.render_loss(out, x, hetero=True))
    s = net.sample(y); assert s.shape == (2, 2, 128, 128) and s.min() >= 0 and s.max() <= 1
    st = renderer.noise_stats(x.numpy(), s.numpy())
    assert 0 <= st["ks_amp_O"] <= 1 and st["active_real_O"] > 0
    # НЗ-маска с классом MH (5) → канал F2 через MH2F2
    yo = torch.tensor([[[0, 5, 1]]]); assert renderer.MH2F2[yo].tolist() == [[[0, 1, 1]]]


def test_oblique_muf_readouts_match_labels(sao_ji):
    from pyon.oblique_train import muf_readouts
    ys, labs = zip(*[obs.oblique_masks_from_sao(sao_ji, d, "O") for d in obs.D_SET])
    f1, f2, pn = muf_readouts(np.stack(ys))
    for k, lab in enumerate(labs):
        assert abs(f1[k] - lab["muf_F2"]) <= 0.2 and abs(f2[k] - lab["muf_MH"]) <= 0.2   # шаг решётки 0.17 МГц
        assert f2[k] < f1[k] and pn[k] >= obs.P_MIN


def test_hF2_above_foF1():
    y = np.zeros((canon.NH, canon.NF), np.int8)
    def band(cls, f0, f1, h):                       # след толщиной 3 строки (ридаут требует ≥2 пикселей в колонке)
        jf = canon.to_grid(np.linspace(f0, f1, 40), canon.F_MIN, canon.F_MAX, canon.NF)
        jh = canon.to_grid([h], canon.H_MIN, canon.H_MAX, canon.NH)[0]
        y[jh - 1:jh + 2, jf] = canon.CLASSES.index(cls)
    band("F1", 2.0, 4.0, 180); band("F2", 4.1, 7.0, 340); band("F2", 2.0, 2.3, 190)   # «F2» ниже каспа F1
    r = scaler.scale_vertical(y, None, f_b=1.3)
    assert abs(r["hF2"] - 340) < 6 and abs(r["hF"] - 180) < 6 and abs(r["foF1"] - 4.0) < 0.12


def test_sim2real_helpers():
    """Сим2реал (Э3 §3.4): кратники во входе рендерера, трансплантация фона, коррелированный спекл."""
    from pyon import training as T
    y = torch.zeros(2, canon.NH, canon.NF, dtype=torch.long); y[:, 40, 30:60] = 1; y[:, 10, 20:40] = 4
    y2 = T.add_multiples(y, 1.0)
    step = (canon.H_MAX - canon.H_MIN) / (canon.NH - 1); off = round(canon.H_MIN / step)
    assert torch.equal(y2[:, 40], y[:, 40]) and (y2[:, 2 * 40 + off, 30:60] == 1).all() and (y2[:, 2 * 10 + off, 20:40] == 4).all()
    assert torch.equal(T.add_multiples(y, 0.0), y)                                  # prob 0 — без изменений
    x_real = torch.rand(2, 2, canon.NH, canon.NF); x_r = torch.ones(2, 2, canon.NH, canon.NF)
    xo = T.transplant(x_real, x_r, y, "own", dilate=2)
    assert (xo[:, :, 38:43, 30:60] == 1).all() and torch.equal(xo[:, :, 60:, :], x_real[:, :, 60:, :])   # внутри — рендер, снаружи — фон
    xs = T.transplant(x_real, x_r, y, "shuffle", dilate=2)
    assert (xs[:, :, 38:43, 30:60] == 1).all() and xs.shape == x_real.shape
    net = renderer.Renderer(base=8, depth=2, hetero=True)
    p = torch.full((8, canon.NH, canon.NF), 0.3)
    for corr, want_corr in (((0, 0), False), ((9, 3), True)):
        net.corr = corr; torch.manual_seed(0)
        act = (net._uniform(p) < p).float()
        assert abs(act.mean().item() - 0.3) < 0.01                                  # маргинал сохранён
        lag = (act[:, 1:] * act[:, :-1]).mean().item() / act.mean().item() ** 2      # соседи по высоте
        assert (lag > 1.5) == want_corr
