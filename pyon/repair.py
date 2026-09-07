# -*- coding: utf-8 -*-
"""
repair.py — онтология НА ИНФЕРЕНСЕ: исправление предсказанной разметки по сработавшему SHACL-гейту.

Основание (E2, 2026-09-06): гейт срабатывает редко (1.3–3.9 % ионограмм), но почти всегда по делу —
как детектор ошибок СТРУКТУРЫ слоёв он даёт precision 0.69 (lognorm) … 0.90 (baseline) при базовой
доле 0.24–0.28. У отбракованных ложный слой F1 встречается в 56–73 % против 15–17 % у принятых,
|Δh′F| медиана 26 км против 6 км. Значит по факту срабатывания разумно не «звать оператора», а
применить минимальное исправление и перемерить характеристики — это и есть третий режим работы
онтологии (обучение → инференс → отчёт), сравниваемый с «без онтологии» и «онтология в лоссе».

Стратегия: перебираем упорядоченный список кандидатов-исправлений (от менее к более радикальным),
пересобираем сцену и валидируем; принимаем ПЕРВОЕ, снимающее все нарушения. Если ни одно не
помогает — возвращаем исходную разметку с пометкой (для отчёта это «гейт сработал, починить не
удалось»). Порядок кандидатов задан находками E2: сначала снимаем слой F1 (главный источник),
затем E/Es, на НЗ — кратник MH (его чаще всего путают с X-следом).

Запуск оценки: python -m pyon.repair --weights runs/E1/lognorm/weights.pt --manifest data/manifest_e1.csv --n 600
                python -m pyon.repair --kind oblique --weights runs/E5/lognorm/weights.pt --dataset data/oblique3 --n 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import canon, gates, scaler                                        # noqa: E402
from pyon import oblique_synth as obs                                        # noqa: E402
from pyon import validate as vd                                              # noqa: E402

VS_CANDIDATES = [("drop", "F1"), ("drop", "E"), ("drop", "Es"), ("drop", "F1E")]
# НЗ-сцена (gates.OB_MODES) строится ТОЛЬКО из F2, Es и MH — значит "drop F1" и "drop E" были
# холостыми и не могли починить ничего (разбор форм 2026-09-07: из 26 отбраковок на трёх ранах
# 18 — форма S4 «задержка Es не меньше задержки F2», а действия снять Es в наборе не было).
OB_CANDIDATES = [("to_x", "MH"), ("drop", "Es"), ("drop", "MH"), ("drop", "EsMH")]


def _apply(pm: np.ndarray, action, classes) -> np.ndarray:
    kind, arg = action
    out = pm.copy()
    if kind == "drop":
        for c in ([arg] if arg in classes else [arg[:2], arg[2:]]):
            if c in classes:
                out[out == classes.index(c)] = 0
    elif kind == "to_x":                       # переназначить класс в X (необыкновенная волна)
        if arg in classes and "X" in classes:
            out[out == classes.index(arg)] = classes.index("X")
    return out


def repair(pm: np.ndarray, kind: str, vocab, gyro: float = 1.3, name: str = "r", known_bad: bool | None = None):
    """→ (исправленная разметка, что применено или None, было ли нарушение).
    known_bad: результат предварительной проверки гейтом (чтобы не валидировать исходную сцену дважды —
    массовую проверку дешевле сделать пулом процессов через `gates.gate_rate`)."""
    classes = canon.CLASSES if kind == "vertical" else obs.OB_CLASSES
    scene_fn = gates.vertical_scene if kind == "vertical" else gates.oblique_scene
    cands = VS_CANDIDATES if kind == "vertical" else OB_CANDIDATES

    def bad(mask, tag):
        scene = scene_fn(mask, tag, gyro) if kind == "vertical" else scene_fn(mask, tag)
        return bool(vd.validate_scene(scene, vocab, verbose=False)["violations"])

    if known_bad is None:
        known_bad = bad(pm, name + "_0")
    if not known_bad:
        return pm, None, False
    for k, act in enumerate(cands):
        cand = _apply(pm, act, classes)
        if np.array_equal(cand, pm):
            continue
        if not bad(cand, f"{name}_{k + 1}"):
            return cand, act, True
    return pm, None, True


def evaluate_vertical(weights: str, manifest: str, n: int, workers: int = 3, dev="cuda"):
    from pyon import training as T
    from pyon.external_test import load_net
    net, c = load_net(ROOT / weights, torch.device(dev))
    df = pd.read_csv(ROOT / manifest, low_memory=False)
    va = df[df.split == "val"]
    sel = va.iloc[np.unique(np.linspace(0, len(va) - 1, min(n, len(va))).round().astype(int))].reset_index(drop=True)
    X, Y, P = T.decode(sel, workers)
    pm, _, prof = T.predict(net, X, torch.device(dev), profile=c.get("profile", False))
    vocab = vd.load_vocabulary(); gy = T.gyros_of(sel)
    t0 = time.time()
    _, _, flags0 = gates.gate_rate(pm, gates.vertical_scene, vocab, prefix="chk_", procs=3,
                                   with_warnings=True, gyros=gy)                # массовая проверка пулом
    rep, acts, flags = [], [], list(map(bool, flags0))
    for i in range(len(pm)):
        r, act, _ = repair(pm[i], "vertical", vocab, float(gy[i]), f"v{i}", known_bad=flags[i])
        rep.append(r); acts.append(act)
    rep = np.stack(rep)
    ct0 = T.char_table(pm, prof, sel); ct1 = T.char_table(rep, prof, sel)
    out = {"weights": str(weights), "n": len(pm), "flagged": float(np.mean(flags)),
           "repaired": float(np.mean([a is not None for a in acts])), "time_s": time.time() - t0}
    for name in ("foF2", "foF1", "hF", "hF2", "MUF3000"):
        for tag, ct in (("before", ct0), ("after", ct1)):
            st = T.err_stats(ct[f"{name}_pred"], ct[f"{name}_artist"])
            out[f"{name}/{tag}_rmse"] = st.get("rmse", np.nan); out[f"{name}/{tag}_med"] = st.get("med", np.nan)
            out[f"{name}/{tag}_n"] = st.get("n", 0)
    fl = np.array(flags)
    if fl.any():                                # только по отбракованным — там и должен быть эффект
        for name in ("foF2", "hF"):
            for tag, ct in (("before", ct0), ("after", ct1)):
                st = T.err_stats(ct[f"{name}_pred"][fl], ct[f"{name}_artist"][fl])
                out[f"{name}/flagged_{tag}_rmse"] = st.get("rmse", np.nan); out[f"{name}/flagged_{tag}_med"] = st.get("med", np.nan)
    from collections import Counter
    out["actions"] = dict(Counter(str(a) for a in acts if a is not None))
    return out


def evaluate_oblique(weights: str, dataset: str, n: int, renderer: str, dev="cuda"):
    from pyon import loader, oblique_train as OT, renderer as rnd
    from pyon.models import UNet
    ck = torch.load(ROOT / weights, map_location=dev); c = ck["cfg"]
    net = UNet(2, len(obs.OB_CLASSES), base=c["base"], depth=c["depth"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    ren = rnd.load_renderer(ROOT / (renderer or c["renderer"]), dev); mh2f2 = rnd.MH2F2.to(dev)
    vds = loader.ObliqueShardDataset(ROOT / dataset, "val", cache=1)
    pick = np.unique(np.linspace(0, len(vds) - 1, min(n, len(vds))).round().astype(int))
    pick = pick[np.argsort([vds.shard_of(int(i)) for i in pick], kind="stable")]
    items = [vds[int(i)] for i in pick]
    Yo = torch.stack([a for a, _, _ in items]); Yx = torch.stack([b for _, b, _ in items])
    L = torch.stack([c_ for _, _, c_ in items]).numpy()
    torch.manual_seed(0)
    X = torch.cat([OT.render_input(ren, mh2f2, Yo[k:k + 128].to(dev).long(), Yx[k:k + 128].to(dev).long(),
                                   "ox", True, 0.7, (0.03, 0.18)).cpu() for k in range(0, len(Yo), 128)])
    pm, _ = OT.predict(net, X, dev)
    vocab = vd.load_vocabulary()
    t0 = time.time()
    _, _, flags0, det = gates.gate_rate(pm, gates.oblique_scene, vocab, prefix="chk_", procs=3,
                                        with_warnings=True, with_details=True)
    rep, acts, flags = [], [], list(map(bool, flags0))
    for i in range(len(pm)):
        r, act, _ = repair(pm[i], "oblique", vocab, name=f"o{i}", known_bad=flags[i])
        rep.append(r); acts.append(act)
    rep = np.stack(rep)
    from collections import Counter as _C
    out = {"weights": str(weights), "n": len(pm), "flagged": float(np.mean(flags)),
           "repaired": float(np.mean([a is not None for a in acts])), "time_s": time.time() - t0,
           "shapes": dict(_C(sh for d in det for sh in set(d)))}
    from pyon import training as T
    for tag, m in (("before", pm), ("after", rep)):
        f1, f2, _ = OT.muf_readouts(m)
        for name, pred, lab in (("MUF1F2", f1, L[:, 0]), ("MUF2F2", f2, L[:, 4])):
            st = T.err_stats(pred, lab)
            out[f"{name}/{tag}_rmse"] = st.get("rmse", np.nan); out[f"{name}/{tag}_med"] = st.get("med", np.nan)
    from collections import Counter
    out["actions"] = dict(Counter(str(a) for a in acts if a is not None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="vertical", choices=["vertical", "oblique"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", default="data/manifest_e1.csv")
    ap.add_argument("--dataset", default="data/oblique3")
    ap.add_argument("--renderer", default="")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = (evaluate_vertical(a.weights, a.manifest, a.n, a.workers) if a.kind == "vertical"
         else evaluate_oblique(a.weights, a.dataset, a.n, a.renderer))
    print(json.dumps(r, ensure_ascii=False, indent=1))
    out = ROOT / (a.out or f"runs/repair_{a.kind}_{Path(a.weights).parent.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True); json.dump(r, open(out, "w"), ensure_ascii=False, indent=1)
    print("→", out)


if __name__ == "__main__":
    main()
