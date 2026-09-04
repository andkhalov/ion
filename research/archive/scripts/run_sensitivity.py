#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sensitivity.py — зависимость доли допустимых разметок от размера словаря мод.

Основной прогон (run_validation.py) выполнен при словаре из семи мод. Возникает
вопрос, не является ли полученное сужение пространства артефактом этого выбора.
Здесь та же модельная сцена из четырёх треков проверяется при словарях из
5, 6, 7, 8 и 9 мод; словарь расширяется вложенно, порядок добавления фиксирован
и задан заранее (моды перечислены по возрастанию кратности и по слоям).

Набор ограничений во всех прогонах один и тот же: S1–S6.

Запуск: local/venv/bin/python research/run_sensitivity.py
Результат дописывается в research/sensitivity-log.txt
"""
import time
from itertools import product
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
    "3F2p": (3, "Layer_F2", "upperRay"),
    "2E":   (2, "Layer_E",  "lowerRay"),
}
ORDER = ["1F2", "1F2p", "2F2", "2F2p", "3F2", "1E", "1Es", "3F2p", "2E"]
SCENE = [("tr1", "10.1", "3.67"), ("tr2", "10.1", "4.10"),
         ("tr3", "7.2", "4.69"), ("tr4", "17.7", "3.34")]
# Es намеренно отсутствует: форма S4 на него не распространяется (см. iono-core.ttl)
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

    say("Зависимость доли допустимых разметок от размера словаря мод")
    say("Сцена — та же (четыре трека), набор ограничений — S1–S6.")
    say(f"Порядок расширения словаря: {', '.join(ORDER)}")
    say("")
    say(f"{'|V|':>4} {'словарь':<44} {'всего':>7} {'прошло':>7} {'доля, %':>9} {'истинная':>9}")
    for k in range(5, 10):
        vocab = ORDER[:k]
        total = passed = 0
        true_ok = False
        t0 = time.time()
        for combo in product(vocab, repeat=4):
            g = scene_graph(combo, base)
            conforms, _, _ = validate(g, shacl_graph=shapes, inference="none",
                                      advanced=True, debug=False)
            total += 1
            if conforms:
                passed += 1
                if combo == TRUE:
                    true_ok = True
        say(f"{k:>4} {','.join(vocab):<44} {total:>7} {passed:>7} "
            f"{100*passed/total:>9.2f} {'да' if true_ok else 'НЕТ':>9}"
            f"   [{time.time()-t0:.0f} с]")

    (HERE / "sensitivity-log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say(f"\nлог записан: {HERE / 'sensitivity-log.txt'}")


if __name__ == "__main__":
    main()
