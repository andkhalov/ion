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
    net = renderer.Renderer(base=8, depth=2)
    y = torch.zeros(2, 128, 128, dtype=torch.long); y[:, 40:43, 20:60] = 1
    inp = net.make_input(y)
    assert inp.shape == (2, renderer.N_CLASSES + 3, 128, 128) and inp[:, -1].min() >= 0 and inp[:, -1].max() <= 1
    out = net(y); assert out.shape == (2, 4, 128, 128)
    x = torch.rand(2, 2, 128, 128) * (torch.rand(2, 2, 128, 128) > 0.9)
    loss = renderer.render_loss(out, x); assert torch.isfinite(loss) and loss > 0
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
