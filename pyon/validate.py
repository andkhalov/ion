#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate.py — единственная правильная точка входа для проверки сцен (Ч-04).

Порядок (обязателен, см. README):
  1) словарь + сцена в один граф;
  2) owlrl.DeductiveClosure(OWLRL_Semantics) — БЕЗ этого шага формы молча
     дают conforms=True на заведомо неверных данных (reflectsFromLayer
     выводится цепочкой, VerticalIonogram — из сеанса);
  3) pyshacl c формами ГИГИЕНЫ (H0–H5); при нарушениях H — стоп
     (ill-typed литерал роняет арифметику форм, T-01/T-02);
  4) pyshacl с содержательными формами (S, V, Q), advanced=True.

Вердикт по sh:Violation; sh:Warning — отдельным списком (в SHACL
conforms=false при любых результатах, поэтому считаем сами).

Запуск:  python validate.py                 — регрессионный набор
         python validate.py scene.ttl [...] — проверка своих сцен

Числовые литералы в сценах строить ТОЛЬКО так (T-01):
    from validate import dec
    Literal(dec(8.2))   # или dec(8.2) напрямую — это уже rdflib.Literal
"""
import sys
from decimal import Decimal
from pathlib import Path

import owlrl
import pyshacl
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

HERE = Path(__file__).resolve().parent
PYONTO = HERE.parent / "pyonto"   # онтология — замкнутая директория pyonto/
IONO = Namespace("http://scilibai.ru/onto/iono#")
SH = Namespace("http://www.w3.org/ns/shacl#")

VOCAB_FILES = ["iono-core.ttl", "iono-char.ttl", "iono-quality.ttl",
               "iono-es.ttl", "iono-phenomena.ttl", "iono-observation.ttl"]
HYGIENE_FILES = ["iono-shapes-hygiene.ttl"]
MAIN_SHAPE_FILES = ["iono-shapes.ttl", "iono-shapes-ext.ttl",
                    "iono-shapes-quality.ttl", "iono-shapes-vs.ttl"]


def dec(v) -> Literal:
    """Единственный допустимый конструктор числового литерала (T-01)."""
    return Literal(str(Decimal(str(float(v)))), datatype=XSD.decimal)


def load_graph(files) -> Graph:
    g = Graph()
    for f in files:
        g.parse(PYONTO / f)
    return g


def load_vocabulary() -> Graph:
    return load_graph(VOCAB_FILES)


def _results(report_graph):
    out = {"Violation": [], "Warning": [], "Info": []}
    for r in report_graph.subjects(RDF.type, SH.ValidationResult):
        sev = str(report_graph.value(r, SH.resultSeverity) or "").split("#")[-1] or "Violation"
        msg = str(report_graph.value(r, SH.resultMessage) or "")
        out.setdefault(sev, []).append(msg)
    return out


def validate_scene(data: Graph, vocab: Graph | None = None, verbose: bool = True):
    """Трёхстадийный конвейер (T-01/T-02). Возвращает dict: hygiene_ok, violations, warnings."""
    vocab = vocab if vocab is not None else load_vocabulary()
    hygiene = load_graph(HYGIENE_FILES)
    g = data + vocab

    # Стадия A: гигиена ДО замыкания (ill-typed литерал роняет и owlrl, и формы).
    # H5-предупреждения на этой стадии игнорируем (reflectsFromLayer ещё не выведен).
    _, rep_a, _ = pyshacl.validate(data_graph=g, shacl_graph=hygiene,
                                   inference="none", advanced=True)
    ra = _results(rep_a)
    result = {"hygiene_ok": not ra["Violation"], "hygiene": ra,
              "violations": [], "warnings": []}
    if ra["Violation"]:
        if verbose:
            print("  ГИГИЕНА НЕ ПРОЙДЕНА — замыкание и содержательные формы не запускались:")
            for m in sorted(set(ra["Violation"])):
                print("    H✗", m[:110])
        return result

    # Стадия B: замыкание (страховочный try — чужие ill-typed типы, не покрытые H0)
    try:
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    except Exception as e:
        result["hygiene_ok"] = False
        result["hygiene"]["Violation"].append(f"H!: owlrl упал на замыкании: {type(e).__name__}: {e}")
        if verbose:
            print("  ЗАМЫКАНИЕ НЕ ВЫПОЛНЕНО:", type(e).__name__, str(e)[:90])
        return result
    _, rep_b, _ = pyshacl.validate(data_graph=g, shacl_graph=hygiene,
                                   inference="none", advanced=True)
    rb = _results(rep_b)

    # Стадия C: содержательные формы
    _, rep, _ = pyshacl.validate(data_graph=g, shacl_graph=load_graph(MAIN_SHAPE_FILES),
                                 inference="none", advanced=True)
    r = _results(rep)
    result["violations"] = r["Violation"] + rb["Violation"]
    result["warnings"] = r["Warning"] + rb["Warning"]
    if verbose:
        v, w = sorted(set(result["violations"])), sorted(set(result["warnings"]))
        print(f"  допустима: {not v} | нарушений {len(v)}, предупреждений {len(w)}")
        for m in v:
            print("    ✗", m[:110])
        for m in w:
            print("    ⚠", m[:110])
    return result


# ---------------------------------------------------------------------------
#  Регрессионный набор (по сценариям пяти ревью)
# ---------------------------------------------------------------------------

def _mode(g, name, label, hop, layer, comp, mof, h_km, ray="lowerRay"):
    tr, mo = IONO[name + "_tr"], IONO[name + "_m"]
    ion = IONO[name.split("__")[0]]
    for t in [(tr, RDF.type, IONO.IonogramTrace), (tr, IONO.isTraceIn, ion), (tr, IONO.denotes, mo),
              (mo, RDF.type, IONO.PropagationMode),
              (mo, IONO.hasModeLabel, Literal(label)),
              (mo, IONO.hasHopCount, Literal(int(hop), datatype=XSD.integer)),
              (mo, IONO.hasRayType, IONO[ray]),
              (mo, IONO.hasMagnetoionicComponent, IONO[comp]),
              (mo, IONO.reflectsFromLayer, IONO["Layer_" + layer]),
              (mo, IONO.hasMOFValue, dec(mof)),
              (mo, IONO.hasMinDelayValue, dec(2 * float(h_km) / 299.792458))]:
        g.add(t)


def _char(g, ion_name, name, cls, layer, value=None, comp="ordinary",
          err=None, qletters=(), dletters=(), cause=None, acc=None):
    d, ion = IONO[name], IONO[ion_name]
    g.add((d, RDF.type, IONO[cls]))
    g.add((d, IONO.isCharacteristicOf, ion))
    g.add((d, IONO.refersToLayer, IONO["Layer_" + layer]))
    g.add((d, IONO.refersToComponent, IONO[comp]))
    if value is not None:
        g.add((d, IONO.hasNumericValue, dec(value)))
        g.add((d, IONO.hasUnit, Literal("МГц")))
    if err is not None:
        g.add((d, IONO.hasErrorEstimate, dec(err)))
    for q in qletters:
        g.add((d, IONO.hasQualifyingLetter, IONO[q]))
    for dl in dletters:
        g.add((d, IONO.hasDescriptiveLetter, IONO[dl]))
    if cause:
        g.add((d, IONO.hasNoValueCause, IONO[cause]))
    if acc:
        g.add((d, IONO.hasAccuracyClass, IONO[acc]))


def _vion(g, name):
    ion = IONO[name]
    g.add((ion, RDF.type, IONO.VerticalIonogram))
    g.add((ion, IONO.hasGyroFrequency, dec(1.3)))
    return ion


def regression():
    vocab = load_vocabulary()
    ok = True

    def case(title, scene, expect_viol, expect_hyg=True, expect_warn=None):
        nonlocal ok
        print(f"== {title}")
        r = validate_scene(scene, vocab)
        got_v = sorted({m.split(":")[0] for m in r["violations"]})
        good = (r["hygiene_ok"] == expect_hyg) and (got_v == sorted(expect_viol))
        if expect_warn is not None:
            got_w = sorted({m.split(":")[0] for m in r["warnings"]})
            good = good and all(w in got_w for w in expect_warn)
        print(f"   -> {'PASS' if good else 'FAIL'} (нарушения {got_v}, ожидалось {sorted(expect_viol)})")
        ok = ok and good

    # 1) эталоны тезисов (НЗ): ВЗ-формы не должны их трогать (Ч-01/T-03)
    for name, expect in [("ground-A.ttl", []), ("ground-B.ttl", ["S1", "S2", "S6"]),
                         ("ground-C.ttl", ["S3"])]:
        g = Graph(); g.parse(PYONTO / name)
        case(f"НЗ-эталон {name}", g, expect)

    # 2) корректная ВЗ-сцена: 1F2 O/X + кратники (через сеанс — проверка вывода VerticalIonogram)
    g = Graph(); ion = IONO["vsA"]
    s = IONO["vsA_sess"]; g.add((s, RDF.type, IONO.VerticalSounding)); g.add((s, IONO.producesIonogram, ion))
    g.add((ion, RDF.type, IONO.Ionogram)); g.add((ion, IONO.hasGyroFrequency, dec(1.3)))
    _mode(g, "vsA__1", "1F2", 1, "F2", "ordinary", 8.2, 223)
    _mode(g, "vsA__1x", "1F2x", 1, "F2", "extraordinary", 8.86, 228)   # fo²=fx(fx−fB): fx≈8.86
    _mode(g, "vsA__2", "2F2", 2, "F2", "ordinary", 8.2, 2 * 223)
    _mode(g, "vsA__3", "3F2", 3, "F2", "ordinary", 8.2, 3 * 223)
    case("ВЗ корректная (VerticalIonogram выведен из сеанса)", g, [])

    # 3) ВЗ: перестановка 1F2/2F2 → V3 (+S2)
    g = Graph(); _vion(g, "vsB")
    _mode(g, "vsB__1", "1F2", 1, "F2", "ordinary", 8.2, 2 * 223)
    _mode(g, "vsB__2", "2F2", 2, "F2", "ordinary", 8.2, 223)
    case("ВЗ кратник перепутан", g, ["S2", "V3"])

    # 4) ВЗ: X сдвинут → V4
    g = Graph(); _vion(g, "vsC")
    _mode(g, "vsC__1", "1F2", 1, "F2", "ordinary", 8.2, 223)
    _mode(g, "vsC__1x", "1F2x", 1, "F2", "extraordinary", 10.7, 228)
    case("ВЗ неверное O/X-расщепление", g, ["V4"])

    # 5) numpy-литерал → H0 (гигиена), содержательные формы не запускаются
    import numpy as np
    g = Graph(); _vion(g, "vsN")
    _mode(g, "vsN__1", "1F2", 1, "F2", "ordinary", 8.2, 223)
    g.remove((IONO["vsN__1_m"], IONO.hasMOFValue, None))
    g.add((IONO["vsN__1_m"], IONO.hasMOFValue, Literal(np.float64(8.25), datatype=XSD.decimal)))
    case("numpy.float64 в литерале", g, [], expect_hyg=False)

    # 6) ill-typed decimal → H0, без RecursionError
    g = Graph(); _vion(g, "vsI")
    _mode(g, "vsI__1", "1F2", 1, "F2", "ordinary", 8.2, 223)
    g.add((IONO["vsI__1_m"], IONO.hasMOFValue, Literal("не измерено", datatype=XSD.decimal)))
    case("ill-typed decimal", g, [], expect_hyg=False)

    # 7) сироты: след без ионограммы, характеристика без привязки → H1/H3
    g = Graph()
    g.add((IONO.orph_tr, RDF.type, IONO.IonogramTrace))
    g.add((IONO.orph_c, RDF.type, IONO.CriticalFrequencyDatum))
    g.add((IONO.orph_c, IONO.hasNumericValue, dec(5.0)))
    case("сироты без привязки", g, [], expect_hyg=False)

    # 8) `a Mode_1F2` без признаков → H5 (Warning), гигиена проходит
    g = Graph(); _vion(g, "vsM")
    tr, mo = IONO.vsM_tr, IONO.vsM_m
    g.add((tr, RDF.type, IONO.IonogramTrace)); g.add((tr, IONO.isTraceIn, IONO.vsM)); g.add((tr, IONO.denotes, mo))
    g.add((mo, RDF.type, IONO.Mode_1F2))
    case("Mode_1F2 без признаков", g, [], expect_warn=["H5"])

    # 9) характеристики: корректная дневная сцена
    g = Graph(); _vion(g, "chA")
    _char(g, "chA", "chA_foE", "CriticalFrequencyDatum", "E", 3.0)
    _char(g, "chA", "chA_foF1", "CriticalFrequencyDatum", "F1", 4.5)
    _char(g, "chA", "chA_foF2", "CriticalFrequencyDatum", "F2", 8.0)
    _char(g, "chA", "chA_foEs", "LimitFrequencyDatum", "Es", 5.0)
    _char(g, "chA", "chA_fbEs", "BlanketingFrequencyDatum", "Es", 2.0)
    _char(g, "chA", "chA_fxI", "TopFrequencyDatum", "F", 8.7)
    case("характеристики корректные", g, [])

    # 10) перепутаны foE/foF2 → Q1
    g = Graph(); _vion(g, "chB")
    _char(g, "chB", "chB_foE", "CriticalFrequencyDatum", "E", 8.0)
    _char(g, "chB", "chB_foF2", "CriticalFrequencyDatum", "F2", 3.0)
    case("порядок критических частот", g, ["Q1"])

    # 11) foEs > foF2 — НЕ нарушение (Es вне порядка по определению, Ф-01)
    g = Graph(); _vion(g, "chC")
    _char(g, "chC", "chC_foF2", "CriticalFrequencyDatum", "F2", 4.0)
    _char(g, "chC", "chC_foEs", "LimitFrequencyDatum", "Es", 7.0)
    case("ночной foEs выше foF2 — допустимо", g, [])

    # 12) условие G без буквы → Q7 (Warning), физика чиста
    g = Graph(); _vion(g, "chD")
    _char(g, "chD", "chD_foE", "CriticalFrequencyDatum", "E", 3.0)
    _char(g, "chD", "chD_foEs", "LimitFrequencyDatum", "Es", 2.5)
    _char(g, "chD", "chD_fbEs", "BlanketingFrequencyDatum", "Es", 2.0)
    case("условие G не оформлено", g, [], expect_warn=["Q7"])

    # 13) Ф-14: foF2 = 10, err = 0.15 — точное по ветке 2 % (НЕ Q5)
    g = Graph(); _vion(g, "chE")
    _char(g, "chE", "chE_foF2", "CriticalFrequencyDatum", "F2", 10.0, err=0.15)
    case("foF2 10 МГц, ε=0,15 — точное по ±2 %", g, [])

    # 14) foF2 = 4, err = 0.15 без U → Q5 (Warning)
    g = Graph(); _vion(g, "chF")
    _char(g, "chF", "chF_foF2", "CriticalFrequencyDatum", "F2", 4.0, err=0.15)
    case("foF2 4 МГц, ε=0,15 без U", g, [], expect_warn=["Q5"])

    # 15) Q8: нет значения и нет объяснения
    g = Graph(); _vion(g, "chG")
    _char(g, "chG", "chG_foF2", "CriticalFrequencyDatum", "F2", None)
    case("значение отсутствует без причины", g, ["Q8"])

    # 16) Q10: буква G с причиной 3 — несогласовано (Warning)
    g = Graph(); _vion(g, "chH")
    _char(g, "chH", "chH_foF2", "CriticalFrequencyDatum", "F2", None,
          dletters=["DL_G"], cause="NVC_3")
    case("буква G при причине 3", g, [], expect_warn=["Q10"])

    print("\n" + ("ВСЕ ТЕСТЫ PASS" if ok else "ЕСТЬ FAIL — см. выше"))
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1:
        vocab = load_vocabulary()
        for f in sys.argv[1:]:
            print(f"== {f}")
            g = Graph(); g.parse(f)
            validate_scene(g, vocab)
    else:
        sys.exit(0 if regression() else 1)
