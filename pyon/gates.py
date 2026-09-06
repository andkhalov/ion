# -*- coding: utf-8 -*-
"""
gates.py — SHACL-гейт на инференсе (Э3 §3.3; Э1 §I.8 «жёсткое ограничение (ii)»):
предсказанная (или эталонная) маска → сцена онтологии → `pyon.validate.validate_scene`.

Сцены строятся из ЖЁСТКИХ ридаутов (canon.fmax_readout / hmin_readout) — так гейт проверяет
ровно то, что увидит пользователь-ионосферщик, а не мягкие величины лосса.

ВЗ-сцена `vertical_scene` (iono_study.ipynb, ячейка 16): для классов F2/F1/E —
CriticalFrequencyDatum, для Es — LimitFrequencyDatum (foEs — предельная, не критическая,
ревью Ф-01) со значением fmax; плюс одно-скачковая O-мода каждого класса с hasMOFValue = fmax
и hasMinDelayValue = 2·h′min/c. Формы, которые при этом работают: Q1–Q3, S4, S5, H*.
НЗ-сцена `oblique_scene` (ячейка 31): моды 1F2 (класс F2), 1Es (Es), 2F2 (MH) с МПЧ по оси
fob_axis и задержкой P′min/c. Формы: S1, S2, S4, S5, H*.

`gate_rate(masks, scene_fn, vocab)` → (доля сцен с sh:Violation, флаги по сценам).
Референсы Э3 §3.3: ARTIST-маски и аналитические НЗ-метки обязаны давать 0 %.
"""
from __future__ import annotations

import math

import numpy as np
from rdflib import Graph, Literal, RDF
from rdflib.namespace import XSD

from pyon import canon
from pyon import oblique_synth as obs
from pyon import validate as vd

IONO = vd.IONO
VS_KIND = {"F2": ("CriticalFrequencyDatum", "F2"), "F1": ("CriticalFrequencyDatum", "F1"),
           "E": ("CriticalFrequencyDatum", "E"), "Es": ("LimitFrequencyDatum", "Es")}
OB_MODES = (("F2", "F2", 1), ("Es", "Es", 1), ("MH", "F2", 2))   # (класс маски, слой, скачков)


def _add_mode(g: Graph, ion, tr, mo, hop: int, layer: str, mof: float, delay_ms: float,
              comp: str = "ordinary"):
    for t in [(tr, RDF.type, IONO.IonogramTrace), (tr, IONO.isTraceIn, ion), (tr, IONO.denotes, mo),
              (mo, RDF.type, IONO.PropagationMode),
              (mo, IONO.hasHopCount, Literal(int(hop), datatype=XSD.integer)),
              (mo, IONO.hasRayType, IONO.lowerRay),
              (mo, IONO.hasMagnetoionicComponent, IONO[comp]),
              (mo, IONO.reflectsFromLayer, IONO["Layer_" + layer]),
              (mo, IONO.hasMOFValue, vd.dec(mof)),
              (mo, IONO.hasMinDelayValue, vd.dec(delay_ms))]:
        g.add(t)


def vertical_scene(pm: np.ndarray, name: str, gyro: float = 1.3) -> Graph:
    """ВЗ-маска [NH, NF] (классы canon.CLASSES) → сцена VerticalIonogram."""
    g = Graph()
    ion = IONO["vs_" + name]
    g.add((ion, RDF.type, IONO.VerticalIonogram))
    g.add((ion, IONO.hasGyroFrequency, vd.dec(gyro)))
    for cls, (dcls, layer) in VS_KIND.items():
        ci = canon.CLASSES.index(cls)
        fo, hm = canon.fmax_readout(pm, ci), canon.hmin_readout(pm, ci)
        if not math.isfinite(fo):
            continue
        d = IONO[f"vs_{name}_c{ci}"]
        for t in [(d, RDF.type, IONO[dcls]), (d, IONO.isCharacteristicOf, ion),
                  (d, IONO.refersToLayer, IONO["Layer_" + layer]),
                  (d, IONO.refersToComponent, IONO.ordinary),
                  (d, IONO.hasNumericValue, vd.dec(fo)), (d, IONO.hasUnit, Literal("МГц"))]:
            g.add(t)
        _add_mode(g, ion, IONO[f"vs_{name}_t{ci}"], IONO[f"vs_{name}_m{ci}"], 1, layer, fo,
                  2.0 * hm / obs.C_KM_MS)
    return g


def oblique_scene(pm: np.ndarray, name: str) -> Graph:
    """НЗ-маска [NP, NF] (классы obs.OB_CLASSES) → сцена ObliqueIonogram."""
    g = Graph()
    ion = IONO["ob_" + name]
    g.add((ion, RDF.type, IONO.ObliqueIonogram))
    for cls, layer, hop in OB_MODES:
        ci = obs.OB_CLASSES.index(cls)
        fm = canon.fmax_readout(pm, ci, obs.fob_axis)
        pmn = canon.hmin_readout(pm, ci, obs.p_axis)
        if not math.isfinite(fm):
            continue
        _add_mode(g, ion, IONO[f"ob_{name}_t{ci}"], IONO[f"ob_{name}_m{ci}"], hop, layer, fm,
                  pmn / obs.C_KM_MS)
    return g


_VOCAB: Graph | None = None


def _worker_init():
    global _VOCAB
    _VOCAB = vd.load_vocabulary()


def _check(args):
    scene_kind, pm, name, gyro = args
    scene = vertical_scene(pm, name, gyro if gyro else 1.3) if scene_kind == "vertical" else oblique_scene(pm, name)
    r = vd.validate_scene(scene, _VOCAB, verbose=False)
    return bool(r["violations"]), bool(r["warnings"]), tuple(sorted(set(map(str, r["violations"]))))


def gate_rate(masks, scene_fn, vocab: Graph | None = None, prefix: str = "g", procs: int = 1,
              with_warnings: bool = False, gyros=None, with_details: bool = False):
    """Доля масок, чья сцена даёт sh:Violation, и флаги по маскам (True = нарушение);
    with_warnings=True → (доля нарушений, доля предупреждений sh:Warning, флаги нарушений) —
    Э3 §3.3 требует долю предупреждений отдельно; with_details=True добавляет 4-й элемент — список
    кортежей нарушенных форм по маскам (E2: разбор отбраковок). procs > 1 — пул процессов (каждый
    грузит словарь один раз): ~1.4 с/сцену на ядро (owlrl-замыкание + 3 прогона pyshacl), на 4 ядрах ×2.3."""
    masks = list(masks)
    if not masks:
        empty = (float("nan"), float("nan"), []) if with_warnings else (float("nan"), [])
        return empty + ([],) if with_details else empty
    kind = "vertical" if scene_fn is vertical_scene else "oblique"
    gyros = list(gyros) if gyros is not None else [None] * len(masks)   # гирочастота станции для V4
    jobs = [(kind, pm, f"{prefix}{k}", gyros[k]) for k, pm in enumerate(masks)]
    if procs > 1:
        from multiprocessing import get_context
        with get_context("fork").Pool(procs, initializer=_worker_init) as pool:
            res = pool.map(_check, jobs, chunksize=4)
    else:
        global _VOCAB
        _VOCAB = vocab if vocab is not None else (_VOCAB or vd.load_vocabulary())
        res = [_check(j) for j in jobs]
    viol = [v for v, _, _ in res]; warn = [w for _, w, _ in res]; det = [d for _, _, d in res]
    out = (sum(viol) / len(viol), sum(warn) / len(warn), viol) if with_warnings else (sum(viol) / len(viol), viol)
    return out + (det,) if with_details else out
