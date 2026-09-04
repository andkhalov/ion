#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ground.py — порождение тестовых наборов T_ground для онтологии iono.

Порождает:
  onto/ground-A.ttl   — сцена с физически допустимой разметкой;
  onto/ground-B.ttl   — та же сцена с переставленными метками 1F2 <-> 2F2
                        (визуально неотличима, физически недопустима);
  onto/ground-C.ttl   — верхний луч без нижнего (проверка допущения ММЛ XI-XIV);
  onto/ground-enum.ttl — полный перебор приписываний меток четырём трекам
                        сцены A из словаря семи мод (7^4 = 2401 вариант).

Измеренные величины (МНЧ и минимальная задержка) приписаны ТРЕКАМ и не
меняются при смене разметки: они снимаются с изображения детерминированным
измерителем. Меняется только отношение denotes.

Запуск:  local/venv/bin/python research/build_ground.py
"""
from itertools import product
from pathlib import Path

ONTO = Path(__file__).resolve().parent / "onto"
ONTO.mkdir(exist_ok=True)

HEADER = """@prefix iono: <http://scilibai.ru/onto/iono#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

"""

# --- шаблоны мод: (метка, кратность, слой, тип луча) ---------------------
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

# --- сцена: четыре трека с характеристиками ------------------------------
# Опорная ионограмма — пример РД 52.26.817-2023 (рисунок 32): трасса
# Ловозеро - Горьковская, 01.06.2014, 19:47 UT. Опубликованные значения:
#   задержки 1EEs 3,34 мс и 1F2 3,67 мс                    [РД, §7.3.3]
#   МНЧ: 1F2 = 10,1 МГц, 2F2 = 7,2 МГц, E/Es = 17,7 МГц    [РД, §7.3.5]
# Не опубликованы и приняты для сцены:
#   задержка 2F2   = 4,69 мс  — плоскоземельная оценка P = sqrt(D^2+(2nh)^2)
#   задержка 1F2п  = 4,10 мс  — то же, при большей высоте отражения
#   МНЧ 1F2п = МНЧ 1F2 = 10,1 МГц — лучи сходятся на предельной частоте
SCENE = [  # (трек, МНЧ МГц, минимальная задержка мс)
    ("tr1", "10.1", "3.67"),
    ("tr2", "10.1", "4.10"),
    ("tr3", "7.2",  "4.69"),
    ("tr4", "17.7", "3.34"),
]

LABELLING_A = {"tr1": "1F2", "tr2": "1F2p", "tr3": "2F2", "tr4": "1Es"}
LABELLING_B = {"tr1": "2F2", "tr2": "1F2p", "tr3": "1F2", "tr4": "1Es"}


def mode_block(mid, label, mof, delay, seg_id, extra=""):
    hop, layer, ray = MODES[label]
    return f"""iono:{mid} a owl:NamedIndividual , iono:PropagationMode ;
    iono:hasModeLabel "{label}" ;
    iono:hasHopCount {hop} ;
    iono:hasRayType iono:{ray} ;
    iono:hasPathClass iono:greatCircle ;
    iono:hasMagnetoionicComponent iono:unresolved ;
    iono:hasTrajectorySegment iono:{seg_id} ;
    iono:occursOn iono:Path_P1 ;
    iono:hasMOFValue {mof} ;
    iono:hasMinDelayValue {delay}{extra} .
iono:{seg_id} a owl:NamedIndividual , iono:TrajectorySegment ;
    iono:reflectsFrom iono:{layer} .
"""


def scene_ttl(ion_id, labelling, traces, materialize_chain=False, extra_by_mode=None):
    out = [f"""iono:{ion_id} a owl:NamedIndividual , iono:Ionogram .
"""]
    extra_by_mode = extra_by_mode or {}
    for tr, mof, delay in traces:
        label = labelling[tr]
        tid = f"{ion_id}_{tr}"
        mid = f"{ion_id}_m_{tr}"
        sid = f"{ion_id}_s_{tr}"
        out.append(f"""iono:{tid} a owl:NamedIndividual , iono:IonogramTrace ;
    iono:isTraceIn iono:{ion_id} ;
    iono:denotes iono:{mid} .
""")
        extra = extra_by_mode.get(tr, "")
        out.append(mode_block(mid, label, mof, delay, sid, extra))
        if materialize_chain:
            _, layer, _ = MODES[label]
            out.append(f"iono:{mid} iono:reflectsFromLayer iono:{layer} .\n")
    return "".join(out)


PATH_TTL = """iono:Path_P1 a owl:NamedIndividual , iono:RadioPath ;
    rdfs:label "Условная односкачковая среднеширотная трасса"@ru ;
    iono:hasPathType iono:subauroral .

"""


def main():
    # --- A: допустимая разметка ---
    (ONTO / "ground-A.ttl").write_text(
        HEADER + "# Сцена A: физически допустимая разметка сцены из четырёх треков.\n\n"
        + PATH_TTL + scene_ttl("ion_A", LABELLING_A, SCENE), encoding="utf-8")

    # --- B: перестановка меток 1F2 <-> 2F2 ---
    (ONTO / "ground-B.ttl").write_text(
        HEADER + "# Сцена B: та же геометрия и те же измеренные значения,\n"
        "# метки 1F2 и 2F2 переставлены. Изображение неотличимо от сцены A.\n\n"
        + PATH_TTL + scene_ttl("ion_B", LABELLING_B, SCENE), encoding="utf-8")

    # --- C: верхний луч без нижнего, с допущением и без ---
    scene_c = [("tr1", "10.1", "4.10"), ("tr2", "7.2", "4.69")]
    lab_c = {"tr1": "1F2p", "tr2": "2F2"}
    (ONTO / "ground-C.ttl").write_text(
        HEADER + "# Сцена C1: верхний луч 1F2п размечен без нижнего луча 1F2 -> нарушение S3.\n"
        "# Сцена C2: то же с явным допущением о коротком нижнем луче (ММЛ XI-XIV) -> нарушения нет.\n\n"
        + PATH_TTL
        + scene_ttl("ion_C1", lab_c, scene_c)
        + scene_ttl("ion_C2", lab_c, scene_c,
                    extra_by_mode={"tr1": " ;\n    iono:shortLowerRayAssumption true"}),
        encoding="utf-8")

    # --- перебор: 7^4 приписываний ---
    parts = [HEADER,
             "# Полный перебор приписываний меток четырём трекам сцены A\n"
             f"# из словаря {len(VOCAB)} мод: {len(VOCAB)}^4 = {len(VOCAB)**4} вариантов.\n"
             "# Свойство reflectsFromLayer материализовано явно (результат цепочки\n"
             "# hasTrajectorySegment o reflectsFrom), чтобы перебор не требовал\n"
             "# построения OWL RL-замыкания на каждом варианте.\n\n",
             PATH_TTL]
    keys = [t[0] for t in SCENE]
    n = 0
    for combo in product(VOCAB, repeat=4):
        lab = dict(zip(keys, combo))
        parts.append(scene_ttl(f"ion_e{n:04d}", lab, SCENE, materialize_chain=True))
        n += 1
    (ONTO / "ground-enum.ttl").write_text("".join(parts), encoding="utf-8")
    print(f"порождено сцен перебора: {n}")
    for f in ["ground-A.ttl", "ground-B.ttl", "ground-C.ttl", "ground-enum.ttl"]:
        p = ONTO / f
        print(f"  {f}: {p.stat().st_size/1024:.1f} КБ")


if __name__ == "__main__":
    main()
