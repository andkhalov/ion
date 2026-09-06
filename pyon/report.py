# -*- coding: utf-8 -*-
"""
report.py — сводки по ранам этапа (Э3 §2 критерии перехода, §3.5 треки «модель vs baseline
vs ARTIST», §5 критерии S1–S5) из артефактов runs/<этап>/<ран>/ (summary.json, tracks.csv).

  python -m pyon.report --stage E1 [--which val/CE|last|val/foF2_med] — таблица ранов + критерии E1/S1
  python -m pyon.report --stage E1 --tracks         — figures/<этап>_tracks_<станция>_<дата>.png:
                                                      наложение треков всех ранов и ARTIST
  python -m pyon.report --stage arch --sort val/foF2_med
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COLS = ["val/IoU_F2", "val/IoU_F1", "val/IoU_E", "val/IoU_Es", "val/foF2_rmse", "val/foF2_med",
        "val/foF2_rd_exact", "val/foF2_rd_doubt", "val/fxI_rmse", "val/fmin_rmse", "val/hF_rmse", "val/hmF2_rmse",
        "val/hmF2_med", "val/prof_rmse", "val/prof_valid_iou", "val/logic_total", "val/logic_P1",
        "val/logic_P2", "val/logic_P3", "val/logic_P4", "gate/violations", "gate/warnings", "val/CE"]


def pick_epoch(metrics: pd.DataFrame, by: str) -> pd.Series:
    """Строка metrics.csv: by='last' — последняя эпоха; иначе — эпоха с минимумом метрики by
    (например val/CE, val/foF2_med); NaN игнорируются."""
    if by == "last" or by not in metrics or metrics[by].isna().all():
        return metrics.iloc[-1]
    return metrics.loc[metrics[by].idxmin()]


def stage_table(stage: str, which: str = "val/CE") -> pd.DataFrame:
    """Таблица ранов этапа; which — критерий выбора эпохи ('last' или имя метрики, минимум)."""
    rows = []
    for sj in sorted((ROOT / "runs" / stage).glob("*/summary.json")):
        s = json.loads(sj.read_text(encoding="utf-8"))
        mcsv = sj.parent / "metrics.csv"
        m = pick_epoch(pd.read_csv(mcsv), which).to_dict() if mcsv.exists() else (s.get("last") or {})
        if mcsv.exists():                                   # гейт считается не каждую эпоху — берём последний доступный
            mm = pd.read_csv(mcsv)
            for g in [c for c in mm.columns if c.startswith("gate/")]:
                col = mm[g].dropna()
                if len(col):
                    m[g] = float(col.iloc[-1])
        cfg = json.loads((sj.parent / "config.json").read_text(encoding="utf-8")) if (sj.parent / "config.json").exists() else {}
        r = dict(run=s["run"], variant=s.get("variant"), params=s.get("params"), epoch=int(m.get("epoch", -1)),
                 depth=cfg.get("depth"), base=cfg.get("base"), norm=cfg.get("norm"), dropout=cfg.get("dropout"),
                 skip=cfg.get("skip"), sched=cfg.get("sched", "const"), epochs=cfg.get("epochs"), seed=cfg.get("seed"),
                 t_min=round(s.get("time_total_s", np.nan) / 60, 1))
        r.update({c: m.get(c, np.nan) for c in COLS})
        rows.append(r)
    return pd.DataFrame(rows)


def e1_criteria(tab: pd.DataFrame) -> list[str]:
    """Э3 §2 (E1): L_logic ↓ ≥ 10× против baseline; foF2 RMSE ≤ baseline; гейт ≤ baseline.
    Э3 §5 (S1): IoU_F2 ≥ 0.3, |ΔfoF2| медиана ≤ 0.15 МГц."""
    out = []
    if "baseline" not in set(tab.run):
        return ["нет рана baseline — критерии E1 не проверить"]
    b = tab[tab.run == "baseline"].iloc[0]
    for _, r in tab.iterrows():
        s1 = (r["val/IoU_F2"] >= 0.3) and (r["val/foF2_med"] <= 0.15)
        line = f"{r.run:>10s}: S1 {'✓' if s1 else '✗'} (IoU_F2 {r['val/IoU_F2']:.3f}, foF2 med {r['val/foF2_med']:.3f})"
        if r.run != "baseline":
            ratio = b["val/logic_total"] / max(r["val/logic_total"], 1e-9)
            line += (f" | L_logic ×{ratio:.1f} {'✓' if ratio >= 10 else '✗'}"
                     f" | RMSE {r['val/foF2_rmse']:.3f} vs {b['val/foF2_rmse']:.3f} {'✓' if r['val/foF2_rmse'] <= b['val/foF2_rmse'] else '✗'}"
                     f" | гейт {r['gate/violations']:.1%} vs {b['gate/violations']:.1%} {'✓' if r['gate/violations'] <= b['gate/violations'] else '✗'}")
        out.append(line)
    return out


def track_figures(stage: str, out_dir: Path) -> list[Path]:
    """Наложение суточных треков всех ранов этапа и ARTIST по фиксированным суткам (Э3 §3.5)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = []
    for tc in sorted((ROOT / "runs" / stage).glob("*/tracks.csv")):
        t = pd.read_csv(tc)
        if len(t):
            t["run"] = tc.parent.name; frames.append(t)
    if not frames:
        return []
    T = pd.concat(frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for (st, date, kind), g in T.groupby(["station", "date", "kind"]):
        fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
        ref = g[g.run == g.run.iloc[0]].sort_values("time")
        hours = pd.to_datetime(ref.time).dt.hour + pd.to_datetime(ref.time).dt.minute / 60
        ax.plot(hours, ref.foF2_artist, "o", ms=3, color="black", label="ARTIST", zorder=3)
        for run, gr in g.groupby("run"):
            gr = gr.sort_values("time")
            h = pd.to_datetime(gr.time).dt.hour + pd.to_datetime(gr.time).dt.minute / 60
            ok = np.isfinite(gr.foF2_pred.values)
            rmse = np.sqrt(np.nanmean((gr.foF2_pred.values - gr.foF2_artist.values) ** 2))
            ax.plot(h[ok], gr.foF2_pred.values[ok], "-", lw=1.3, label=f"{run} (RMSE {rmse:.2f})")
        ax.set(xlabel="UT, ч", ylabel="foF2, МГц", xlim=(0, 24), title=f"{st} {date} ({kind})")
        ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
        p = out_dir / f"{stage}_track_{st}_{date}_{kind}.png"
        fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="E1")
    ap.add_argument("--which", default="val/CE", help="эпоха: 'last' или метрика с минимумом (val/CE, val/foF2_med, ...)")
    ap.add_argument("--sort", default="")
    ap.add_argument("--tracks", action="store_true")
    a = ap.parse_args()
    tab = stage_table(a.stage, a.which)
    if a.sort and a.sort in tab:
        tab = tab.sort_values(a.sort)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(tab.round(3).to_string(index=False))
    if a.stage.startswith("E1"):
        print("\n".join(e1_criteria(tab)))
    if a.tracks:
        for p in track_figures(a.stage, ROOT / "figures"):
            print("→", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
