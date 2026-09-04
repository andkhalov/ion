#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baseline.py — разложение сужения и случайный бейзлайн.

Отвечает на два вопроса, которые задаёт рецензент:
  1. Какая часть сужения приходится на чисто комбинаторное требование
     различности обозначений (форма S5), а какая — на физические формы?
  2. Насколько наблюдаемое число выживших отличается от того, что дали бы
     СЛУЧАЙНЫЕ ограничения тех же долей отсева?

Один проход по всем 7^4 = 2401 приписываниям; для каждого варианта
записывается множество сработавших форм, после чего любые комбинации
считаются без повторного вызова движка.

Запуск: local/venv/bin/python research/run_baseline.py
Результат — research/baseline-log.txt
"""
import random
import time
from collections import Counter
from itertools import combinations, product
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, RDF, XSD

HERE = Path(__file__).resolve().parent
ONTO = HERE / "onto"
IONO = Namespace("http://scilibai.ru/onto/iono#")

MODES = {
    "1F2":  (1, "Layer_F2", "lowerRay"),
    "1F2p": (1, "Layer_F2", "upperRay"),
    "2F2":  (2, "Layer_F2", "lowerRay"),
    "2F2p": (2, "Layer_F2", "upperRay"),
    "3F2":  (3, "Layer_F2", "lowerRay"),
    "1E":   (1, "Layer_E",  "lowerRay"),
    "1Es":  (1, "Layer_Es", "lowerRay"),
}
VOCAB = list(MODES.keys())
SCENE = [("tr1", "10.1", "3.67"), ("tr2", "10.1", "4.10"),
         ("tr3", "7.2", "4.69"), ("tr4", "17.7", "3.34")]
LAYER_ORDER = {"Layer_E": 1, "Layer_Es": 1, "Layer_F1": 2, "Layer_F2": 3}
TRUE = ("1F2", "1F2p", "2F2", "1Es")

LOG = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.append(line)


def scene_graph(labelling, base):
    g = Graph()
    g += base
    ion = IONO["ion"]
    g.add((ion, RDF.type, IONO.Ionogram))
    for (tr, mof, delay), label in zip(SCENE, labelling):
        hop, layer, ray = MODES[label]
        t, m = IONO[f"tr_{tr}"], IONO[f"m_{tr}"]
        g.add((t, RDF.type, IONO.IonogramTrace))
        g.add((t, IONO.isTraceIn, ion))
        g.add((t, IONO.denotes, m))
        g.add((m, RDF.type, IONO.PropagationMode))
        g.add((m, IONO.hasModeLabel, Literal(label)))
        g.add((m, IONO.hasHopCount, Literal(hop)))
        g.add((m, IONO.hasRayType, IONO[ray]))
        g.add((m, IONO.reflectsFromLayer, IONO[layer]))
        g.add((m, IONO.hasMOFValue, Literal(mof, datatype=XSD.decimal)))
        g.add((m, IONO.hasMinDelayValue, Literal(delay, datatype=XSD.decimal)))
    return g


def main():
    shapes = Graph()
    shapes.parse(ONTO / "iono-shapes.ttl")
    shapes.parse(ONTO / "iono-shapes-ext.ttl")
    base = Graph()
    for ln, o in LAYER_ORDER.items():
        base.add((IONO[ln], IONO.layerHeightOrder, Literal(o)))

    say("Разложение сужения и случайный бейзлайн")
    say("Сцена — четыре трека, словарь семи мод, 7^4 = 2401 приписывание.")
    say("")

    fired = {}          # вариант -> множество сработавших форм
    t0 = time.time()
    for combo in product(VOCAB, repeat=4):
        g = scene_graph(combo, base)
        _, _, rtext = validate(g, shacl_graph=shapes, inference="none",
                               advanced=True, debug=False)
        tags = set()
        for line in rtext.splitlines():
            s = line.strip()
            if s.startswith("Message: S"):
                tags.add(s[9:11])
        fired[combo] = tags
        if len(fired) % 600 == 0:
            say(f"    ...{len(fired)} вариантов, {time.time()-t0:.0f} с")
    say(f"проход завершён за {time.time()-t0:.0f} с\n")

    ALL = ["S1", "S2", "S3", "S4", "S5", "S6"]
    sizes = {s: sum(1 for v in fired.values() if s in v) for s in ALL}
    say("отсев по формам (число вариантов, на которых форма срабатывает):")
    for s in ALL:
        say(f"  {s}: {sizes[s]:5d}  ({100*sizes[s]/2401:.1f} %)")

    def surv(subset):
        return [c for c, v in fired.items() if not (v & set(subset))]

    say("\nразложение:")
    only5 = surv(["S5"])
    say(f"  только S5 (различность обозначений): выживает {len(only5)} "
        f"= 7*6*5*4 = {7*6*5*4} — чисто комбинаторное требование")
    phys = surv(["S1", "S2", "S3", "S4", "S6"])
    say(f"  только физические формы S1–S4, S6:    выживает {len(phys)}")
    all5 = surv(["S1", "S2", "S3", "S4", "S5"])
    all6 = surv(ALL)
    say(f"  S1–S5:                                выживает {len(all5)}")
    say(f"  S1–S6:                                выживает {len(all6)}")
    say(f"  физические формы поверх S5: {len(only5)} -> {len(all6)} "
        f"(в {len(only5)/max(len(all6),1):.1f} раза)")
    say(f"  истинная разметка среди выживших S1–S6: "
        f"{'да' if TRUE in set(all6) else 'НЕТ'}")

    # --- вклад каждой формы: сколько выживших добавится при её удалении ---
    say("\nвклад отдельной формы (сколько вариантов она отсекает сверх остальных пяти):")
    for s in ALL:
        rest = [x for x in ALL if x != s]
        say(f"  без {s}: выживает {len(surv(rest))} вместо {len(all6)}")

    # --- случайный бейзлайн ---
    say("\nслучайный бейзлайн: шесть случайных подмножеств тех же размеров")
    rng = random.Random(20260731)
    universe = list(fired.keys())
    n_sim = 2000
    res = []
    for _ in range(n_sim):
        killed = set()
        for s in ALL:
            killed |= set(rng.sample(universe, sizes[s]))
        res.append(2401 - len(killed))
    res.sort()
    med = res[n_sim // 2]
    lo, hi = res[int(0.05 * n_sim)], res[int(0.95 * n_sim)]
    say(f"  {n_sim} симуляций: медиана {med}, интервал 5–95 % [{lo}, {hi}]")
    say(f"  наблюдаемое при S1–S6: {len(all6)}")

    res5 = []
    for _ in range(n_sim):
        killed = set()
        for s in ALL[:5]:
            killed |= set(rng.sample(universe, sizes[s]))
        res5.append(2401 - len(killed))
    res5.sort()
    say(f"  то же для пяти форм: медиана {res5[n_sim//2]}, "
        f"интервал [{res5[int(0.05*n_sim)]}, {res5[int(0.95*n_sim)]}]; "
        f"наблюдаемое {len(all5)}")

    # --- примеры выживших, но физически сомнительных разметок ---
    say("\nвыжившие при S1–S6 (все):")
    for c in all6:
        say("   " + ", ".join(c))

    (HERE / "baseline-log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say(f"\nлог записан: {HERE / 'baseline-log.txt'}")


if __name__ == "__main__":
    main()
