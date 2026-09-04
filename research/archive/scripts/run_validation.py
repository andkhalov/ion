#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_validation.py — воспроизводимая проверка прототипа онтологии.

Шаги:
  1. Размер онтологии и форм; построение OWL 2 RL-замыкания (owlrl) на сцене A;
     проверка, что ризонер выводит свойство reflectsFromLayer из цепочки
     hasTrajectorySegment ∘ reflectsFrom и классифицирует моды по ортогональным
     признакам (Mode_1F2, Mode_2F2, Mode_1F2p, Mode_1Es).
  2. SHACL-валидация сцен A, B, C1, C2 движком pyshacl.
  3. Полный перебор 7^4 = 2401 приписываний меток четырём трекам сцены A.
     Каждый вариант проверяется тем же движком pyshacl на отдельном малом
     графе; считается число прошедших и вклад каждой формы.

Запуск:  local/venv/bin/python research/run_validation.py
Вывод сохраняется в research/validation-log.txt
"""
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, RDF, URIRef, XSD
import owlrl

HERE = Path(__file__).resolve().parent
ONTO = HERE / "onto"
IONO = Namespace("http://scilibai.ru/onto/iono#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")

MODE_CLASSES = ["Mode_1F2", "Mode_1F2p", "Mode_2F2", "Mode_2F2p",
                "Mode_3F2", "Mode_1E", "Mode_1Es"]

# словарь мод и сцена — те же, что в build_ground.py
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
# Es намеренно отсутствует: форма S4 на него не распространяется (см. iono-core.ttl)
LAYER_ORDER = {"Layer_E": 1, "Layer_Es": 1, "Layer_F1": 2, "Layer_F2": 3}

LOG = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.append(line)


def load(*files):
    g = Graph()
    for f in files:
        g.parse(ONTO / f, format="turtle")
    return g


# ---------------------------------------------------------------- шаг 1
def step1():
    say("=" * 70)
    say("ШАГ 1. Онтология, формы, ризонер")
    say("=" * 70)
    core = load("iono-core.ttl")
    n_cls = len(set(core.subjects(RDF.type, OWL.Class)))
    n_def = len(set(core.subjects(OWL.equivalentClass, None)))
    n_op = len(set(core.subjects(RDF.type, OWL.ObjectProperty)))
    n_dp = len(set(core.subjects(RDF.type, OWL.DatatypeProperty)))
    say(f"iono-core.ttl : {len(core)} триплетов; классов {n_cls} "
        f"(из них определяемых {n_def}); объектных свойств {n_op}; "
        f"свойств-значений {n_dp}")

    shapes = load("iono-shapes.ttl")
    n_sh = len(list(shapes.subjects(URIRef("http://www.w3.org/ns/shacl#select"), None)))
    say(f"iono-shapes.ttl: {len(shapes)} триплетов; SPARQL-ограничений {n_sh}")

    g = load("iono-core.ttl", "ground-A.ttl")
    before = len(g)
    t0 = time.time()
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    say(f"OWL 2 RL-замыкание сцены A: {before} -> {len(g)} триплетов "
        f"за {time.time()-t0:.2f} с")

    say("\nвыведено ризонером (сцена A):")
    for tr, mof, delay in SCENE:
        m = IONO[f"ion_A_m_{tr}"]
        types = sorted(str(t).split("#")[-1] for t in g.objects(m, RDF.type)
                       if str(t).split("#")[-1] in MODE_CLASSES)
        layer = sorted(str(o).split("#")[-1] for o in g.objects(m, IONO.reflectsFromLayer))
        say(f"  {tr}: класс {types or ['—']}; reflectsFromLayer {layer or ['—']}")
    return g


# ---------------------------------------------------------------- шаг 2
def shacl(data, shapes):
    conforms, _, rtext = validate(data, shacl_graph=shapes, inference="none",
                                  advanced=True, debug=False)
    tags = Counter()
    for line in rtext.splitlines():
        s = line.strip()
        if s.startswith("Message: S"):
            tags[s[9:11]] += 1
    return conforms, tags


def step2(gA):
    say("\n" + "=" * 70)
    say("ШАГ 2. SHACL-валидация именованных сцен")
    say("=" * 70)
    shapes = load("iono-shapes.ttl")

    c, t = shacl(gA, shapes)
    say(f"  A  (допустимая разметка)          : conforms={c}, нарушений {sum(t.values())} {dict(t)}")

    for name, f in [("B", "ground-B.ttl"), ("C", "ground-C.ttl")]:
        g = load("iono-core.ttl", f)
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
        c, t = shacl(g, shapes)
        if name == "B":
            say(f"  B  (1F2 и 2F2 переставлены)       : conforms={c}, нарушений {sum(t.values())} {dict(t)}")
        else:
            say(f"  C1+C2 (верхний луч без нижнего;   : conforms={c}, нарушений {sum(t.values())} {dict(t)}")
            say("        второй — с допущением ММЛ XI–XIV)")


# ---------------------------------------------------------------- шаг 3
def build_scene_graph(labelling, base):
    """Малый граф одной сцены: 4 трека, 4 моды, слои с порядком по высоте.
    Свойство reflectsFromLayer записано явно — это результат цепочки
    hasTrajectorySegment ∘ reflectsFrom, выводимый ризонером на шаге 1."""
    g = Graph()
    g += base
    ion = IONO["ion"]
    g.add((ion, RDF.type, IONO.Ionogram))
    for tr, mof, delay in SCENE:
        label = labelling[tr]
        hop, layer, ray = MODES[label]
        t = IONO[f"tr_{tr}"]
        m = IONO[f"m_{tr}"]
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


def step3(shape_files=("iono-shapes.ttl",), title="базовый набор форм S1–S5"):
    say("\n" + "=" * 70)
    say(f"ШАГ 3. Полный перебор приписываний: 7 мод, 4 трека, 7^4 вариантов")
    say(f"        набор ограничений: {title}")
    say("=" * 70)
    shapes = load(*shape_files)
    base = Graph()
    for ln, order in LAYER_ORDER.items():
        base.add((IONO[ln], IONO.layerHeightOrder, Literal(order)))

    total = 0
    passed = []
    tag_counts = Counter()
    scenes_with_tag = Counter()
    t0 = time.time()
    for combo in product(VOCAB, repeat=4):
        lab = dict(zip([s[0] for s in SCENE], combo))
        g = build_scene_graph(lab, base)
        conforms, tags = shacl(g, shapes)
        total += 1
        if conforms:
            passed.append(combo)
        else:
            for k, v in tags.items():
                tag_counts[k] += v
                scenes_with_tag[k] += 1
        if total % 400 == 0:
            say(f"    ...{total} вариантов, {time.time()-t0:.0f} с")

    say(f"\nпроверено вариантов: {total} за {time.time()-t0:.0f} с")
    for tag in sorted(scenes_with_tag):
        say(f"  {tag}: срабатывает на {scenes_with_tag[tag]} вариантах "
            f"({100*scenes_with_tag[tag]/total:.1f} %), всего {tag_counts[tag]} нарушений")
    say(f"\nПРОШЛО ПРОВЕРКУ: {len(passed)} из {total} "
        f"({100*len(passed)/total:.2f} %)")
    say("выжившие разметки (tr1, tr2, tr3, tr4):")
    for combo in passed:
        say("   " + ", ".join(combo))
    return total, len(passed)


if __name__ == "__main__":
    gA = step1()
    step2(gA)
    if "--quick" not in sys.argv:
        step3()
        step3(("iono-shapes.ttl", "iono-shapes-ext.ttl"),
              "S1–S5 плюс дополнительное ограничение S6 (задержка верхнего луча)")
    (HERE / "validation-log.txt").write_text(
        "\n".join(LOG) + "\n", encoding="utf-8")
    say(f"\nлог записан: {HERE / 'validation-log.txt'}")
