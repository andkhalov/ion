# -*- coding: utf-8 -*-
"""SHACL-гейт (Э3 §3.3, §4 п.5): эталонные сцены онтологии; ARTIST-маски и аналитические
НЗ-метки — 0 % нарушений; сломанные сцены ловятся нужными формами."""
import numpy as np
import pytest
from rdflib import Graph

from pyon import canon, gates
from pyon import oblique_synth as obs
from pyon import validate as vd


def _codes(res):
    return sorted({m.split(":")[0] for m in res["violations"]})


@pytest.mark.parametrize("name,expect", [("ground-A.ttl", []), ("ground-B.ttl", ["S1", "S2", "S6"]),
                                         ("ground-C.ttl", ["S3"])])
def test_ground_scenes(vocab, name, expect):
    g = Graph(); g.parse(vd.PYONTO / name)
    res = vd.validate_scene(g, vocab, verbose=False)
    assert res["hygiene_ok"] and _codes(res) == expect


def test_artist_masks_pass_gate(vocab, sao_ji, sao_ro):
    for k, sao in enumerate((sao_ji, sao_ro)):
        y = canon.masks_from_sao(sao)
        res = vd.validate_scene(gates.vertical_scene(y, f"art{k}"), vocab, verbose=False)
        assert res["hygiene_ok"] and res["violations"] == []


def test_broken_vertical_scene_caught_by_q1(vocab, sao_ji):
    y = canon.masks_from_sao(sao_ji)
    jf = canon.to_grid(np.linspace(2.0, 11.0, 60), canon.F_MIN, canon.F_MAX, canon.NF)
    jh = int(canon.to_grid([110.0], canon.H_MIN, canon.H_MAX, canon.NH)[0])
    y[jh - 1:jh + 2, jf] = canon.CLASSES.index("E")                 # «foE» = 11 МГц > foF2 = 8.2
    res = vd.validate_scene(gates.vertical_scene(y, "brokenQ1"), vocab, verbose=False)
    assert "Q1" in _codes(res)


def test_oblique_labels_pass_and_swap_fails(vocab, sao_ji):
    y, lab = obs.oblique_masks_from_sao(sao_ji, 800.0, "O")
    ok = vd.validate_scene(gates.oblique_scene(y, "lab"), vocab, verbose=False)
    assert ok["hygiene_ok"] and ok["violations"] == []
    iF2, iMH = obs.OB_CLASSES.index("F2"), obs.OB_CLASSES.index("MH")
    sw = y.copy(); sw[y == iF2] = iMH; sw[y == iMH] = iF2
    bad = vd.validate_scene(gates.oblique_scene(sw, "swap"), vocab, verbose=False)
    assert {"S1", "S2"} <= set(_codes(bad))


def test_gate_rate(vocab, sao_ji):
    masks = [obs.oblique_masks_from_sao(sao_ji, d, "O")[0] for d in obs.D_SET]
    rate, flags = gates.gate_rate(masks, gates.oblique_scene, vocab, prefix="r")
    assert rate == 0.0 and flags == [False] * len(masks)
    assert np.isnan(gates.gate_rate([], gates.oblique_scene, vocab)[0])


def test_empty_mask_scene_is_valid(vocab):
    res = vd.validate_scene(gates.vertical_scene(np.zeros((canon.NH, canon.NF), np.int8), "empty"), vocab, verbose=False)
    assert res["hygiene_ok"] and res["violations"] == []


def test_regression_suite_18(capsys):
    """Полный регрессионный набор онтологии (обязателен перед каждым full-прогоном)."""
    assert vd.regression()
