# -*- coding: utf-8 -*-
"""
gate_study.py — E2: верификация онтологии на ВЗ (Э3 §2 E2, §3.3, §5 S5; Э2 §6.0 «гейт-фильтр
не работает на спокойной выборке» — проверяем на ВОЗМУЩЁННЫХ днях).

Что делаем: из val-манифеста берём две группы (возмущённые сутки: Ap ≥ 30 ∨ Kp ≥ 5, и спокойные),
стратифицированно по станциям; прогоняем модель; каждую предсказанную маску — через SHACL-гейт
(`gates.gate_rate`, с деталями нарушенных форм), маски ARTIST — как референс; считаем:
  (1) доли нарушений/предупреждений: модель vs ARTIST, спокойные vs возмущённые, по станциям;
  (2) разбор отбраковок: какие формы нарушаются (счётчик по формам), примеры;
  (3) гипотеза «гейт-фильтр»: |ΔfoF2| у отбракованных vs принятых (медиана, доля грубых ошибок
      |Δ| > max(0.5 МГц, 5 %)), precision/recall гейта как детектора грубых ошибок — отдельно на
      спокойных и возмущённых (S5: выигрыш онтологического контура больше на возмущённых).
Артефакты: runs/E2/<ран>/{metrics.json, samples.csv (по образцам: группа, станция, флаги, формы,
|ΔfoF2|), shapes.csv, png/} + TensorBoard.

Запуск: python -m pyon.gate_study --weights runs/E1/lognorm/weights.pt --manifest data/manifest_e1.csv
        [--n 1000 --procs 4 --run lognorm]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import gates, tblog, training as T                                       # noqa: E402
from pyon import validate as vd                                                     # noqa: E402
from pyon.external_test import load_net                                             # noqa: E402


def pick_groups(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """val: n возмущённых и n спокойных, стратифицированно по станциям (равные доли, где есть)."""
    va = df[df.split == "val"]
    out = []
    for grp, sub in (("disturbed", va[va.disturbed == 1]), ("quiet", va[va.disturbed == 0])):
        per = max(1, n // max(1, sub.station.nunique()))
        parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in sub.groupby("station")]
        s = pd.concat(parts).copy(); s["group"] = grp; out.append(s)
    return pd.concat(out).sort_values(["group", "station", "time"]).reset_index(drop=True)


def gross(d: np.ndarray, fo: np.ndarray) -> np.ndarray:
    """Грубая ошибка foF2: |Δ| > max(0.5 МГц, 5 % foF2) (вдвое шире «сомнительно» по РД 52.26.817)."""
    return np.abs(d) > np.maximum(0.5, 0.05 * fo)


def det_stats(flag: np.ndarray, bad: np.ndarray) -> dict:
    ok = np.isfinite(bad.astype(float)); flag, bad = flag[ok].astype(bool), bad[ok].astype(bool)
    tp = int((flag & bad).sum()); fp = int((flag & ~bad).sum()); fn = int((~flag & bad).sum())
    return dict(n=int(ok.sum()), flagged=int(flag.sum()), gross=int(bad.sum()), tp=tp, fp=fp, fn=fn,
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), base_rate=float(bad.mean()) if len(bad) else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", default="data/manifest_e1.csv")
    ap.add_argument("--run", default="")
    ap.add_argument("--n", type=int, default=1000, help="образцов в каждой группе (возмущённые / спокойные)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = a.run or Path(a.weights).parent.name
    rundir = ROOT / "runs" / "E2" / run; (rundir / "png").mkdir(parents=True, exist_ok=True)
    net, c = load_net(ROOT / a.weights, dev)
    df = pd.read_csv(ROOT / a.manifest, low_memory=False)
    sel = pick_groups(df, a.n, a.seed)
    log = tblog.TBLog(rundir, dict(weights=a.weights, manifest=a.manifest, n=a.n))
    t0 = time.time()
    X, Y, P = T.decode(sel, a.workers)
    pm, comps, prof = T.predict(net, X, dev, profile=c.get("profile", False))
    ct = T.char_table(pm, prof, sel)
    print(f"[E2/{run}] {len(sel)} образцов ({(sel.group == 'disturbed').sum()} возмущённых, "
          f"{(sel.group == 'quiet').sum()} спокойных; станции {sel.station.unique().tolist()}), предсказано за {time.time() - t0:.0f} с", flush=True)
    vocab = vd.load_vocabulary(); gy = T.gyros_of(sel)
    t1 = time.time()
    rate, warn, flags, det = gates.gate_rate(pm, gates.vertical_scene, vocab, prefix="m_", procs=a.procs, with_warnings=True, gyros=gy, with_details=True)
    rate_a, warn_a, flags_a, det_a = gates.gate_rate(Y.numpy(), gates.vertical_scene, vocab, prefix="a_", procs=a.procs, with_warnings=True, gyros=gy, with_details=True)
    print(f"  гейт {len(sel)}×2 сцен за {time.time() - t1:.0f} с: модель {rate:.1%} (предупр. {warn:.1%}), ARTIST {rate_a:.1%} ({warn_a:.1%})", flush=True)
    sel = sel.copy()
    sel["viol"] = np.array(flags, bool); sel["viol_artist"] = np.array(flags_a, bool)
    sel["shapes"] = ["|".join(d) for d in det]; sel["shapes_artist"] = ["|".join(d) for d in det_a]
    sel["logic_total"] = sum(comps.values()) if comps else np.nan
    sel["dfoF2"] = (ct.foF2_pred - ct.foF2_artist).values; sel["foF2_pred"] = ct.foF2_pred.values
    sel["gross"] = gross(sel.dfoF2.values, sel.foF2.astype(float).values)
    m = {}
    for grp in ("disturbed", "quiet"):
        g = sel[sel.group == grp]
        m[f"{grp}/n"] = len(g); m[f"{grp}/viol"] = float(g.viol.mean()); m[f"{grp}/viol_artist"] = float(g.viol_artist.mean())
        d = g.dfoF2.values; ok = np.isfinite(d)
        m[f"{grp}/foF2_rmse"] = float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else np.nan
        m[f"{grp}/foF2_med"] = float(np.median(np.abs(d[ok]))) if ok.any() else np.nan
        m[f"{grp}/foF2_med_flagged"] = float(np.nanmedian(np.abs(g.dfoF2[g.viol]))) if g.viol.any() else np.nan
        m[f"{grp}/foF2_med_passed"] = float(np.nanmedian(np.abs(g.dfoF2[~g.viol]))) if (~g.viol).any() else np.nan
        m[f"{grp}/gross_rate"] = float(np.nanmean(g.gross[ok]))
        m[f"{grp}/gross_rate_flagged"] = float(np.nanmean(g.gross[g.viol & ok])) if (g.viol & ok).any() else np.nan
        m[f"{grp}/gross_rate_passed"] = float(np.nanmean(g.gross[~g.viol & ok])) if (~g.viol & ok).any() else np.nan
        for k, v in det_stats(g.viol.values, g.gross.values).items():
            m[f"{grp}/detector_{k}"] = v
        m[f"{grp}/logic_total"] = float(np.nanmean(g.logic_total))
        for st, gs in g.groupby("station"):
            m[f"{grp}/station_{st}/viol"] = float(gs.viol.mean()); m[f"{grp}/station_{st}/viol_artist"] = float(gs.viol_artist.mean())
            m[f"{grp}/station_{st}/foF2_med"] = float(np.nanmedian(np.abs(gs.dfoF2)))
    shapes = collections.Counter(s for d in det for s in d); shapes_a = collections.Counter(s for d in det_a for s in d)
    pd.DataFrame([dict(shape=k, model=shapes.get(k, 0), artist=shapes_a.get(k, 0)) for k in sorted(set(shapes) | set(shapes_a))]).to_csv(rundir / "shapes.csv", index=False)
    sel.drop(columns=[c_ for c_ in sel.columns if c_ in ("path", "sao")]).to_csv(rundir / "samples.csv", index=False)
    # гистограммы |ΔfoF2| для отбракованных/принятых по группам
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for j, grp in enumerate(("disturbed", "quiet")):
        g = sel[sel.group == grp]
        for fl, lab in ((True, "отбраковано гейтом"), (False, "принято")):
            v = np.abs(g.dfoF2[g.viol == fl].dropna())
            ax[j].hist(np.clip(v, 0, 3), bins=30, range=(0, 3), alpha=0.6, density=True, label=f"{lab} (n={len(v)})")
        ax[j].set_title(f"{grp}: |ΔfoF2|, МГц (модель {m[f'{grp}/viol']:.1%} наруш., ARTIST {m[f'{grp}/viol_artist']:.1%})", fontsize=9)
        ax[j].legend(fontsize=8); ax[j].set_yscale("log")
    plt.tight_layout(); plt.savefig(rundir / "png" / "gate_filter_hist.png", dpi=90); log.w.add_figure("E2/gate_filter_hist", fig, 0); plt.close(fig)
    log.scalars(m, 0); log.row(m, 0)
    json.dump(dict(weights=a.weights, manifest=a.manifest, n=a.n, metrics=m,
                   shapes_model=dict(shapes), shapes_artist=dict(shapes_a)), open(rundir / "metrics.json", "w"), indent=1, ensure_ascii=False)
    for grp in ("disturbed", "quiet"):
        print(f"  {grp:9s}: нарушений модель {m[f'{grp}/viol']:.1%} / ARTIST {m[f'{grp}/viol_artist']:.1%}; foF2 RMSE {m[f'{grp}/foF2_rmse']:.2f} мед {m[f'{grp}/foF2_med']:.3f}; "
              f"|Δ| мед отбракованных {m[f'{grp}/foF2_med_flagged']:.2f} vs принятых {m[f'{grp}/foF2_med_passed']:.2f}; грубых {m[f'{grp}/gross_rate']:.1%} "
              f"(среди отбракованных {m[f'{grp}/gross_rate_flagged']:.1%}, принятых {m[f'{grp}/gross_rate_passed']:.1%}); детектор P {m[f'{grp}/detector_precision']:.2f} R {m[f'{grp}/detector_recall']:.2f}", flush=True)
    print(f"  формы (модель): {dict(shapes.most_common(6))}; (ARTIST): {dict(shapes_a.most_common(4))}")
    print(f"[E2/{run}] готово за {time.time() - t0:.0f} с → {rundir}", flush=True)


if __name__ == "__main__":
    main()
