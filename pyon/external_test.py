# -*- coding: utf-8 -*-
"""
external_test.py — внешний ВЗ-тест обученной модели E1 на станция-годе, НЕ входившем в манифест
обучения (Э3 §4 п.3: независимая эпоха/станция; первый случай — JR055-2020, докачан после старта E1).

Все пары внешнего манифеста (`python -m pyon.manifest --include JR055/2020`) прогоняются весами
рана: метрики Э3 §3 как в `training.evaluate` (сегментация, таблица характеристик против ARTIST,
профиль, страты, гейт на первых gate_n сценах + ARTIST-референс), суточные треки по ВСЕМ суткам
(`tracks.csv` + метрики по суткам) и PNG для суток с максимальным Ap и с максимальной долей
многослойных ионограмм; панели «как у дигизонда» — 8 образцов. Артефакты → runs/<stage>/<run>/
(TensorBoard-события, metrics.json, val_readouts.csv, tracks.csv, png/).

Запуск: python -m pyon.external_test --weights runs/E1/baseline/weights.pt --manifest data/manifest_JR055_2020.csv
        [--stage E1ext --run baseline --gate_n 150 --workers 4 --limit 0]
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
from pyon import canon, tblog, training as T                                   # noqa: E402
from pyon import validate as vd                                                 # noqa: E402
from pyon.models import UNet                                                    # noqa: E402


def load_net(weights: Path, dev):
    ck = torch.load(weights, map_location=dev)
    c = ck["cfg"]
    net = UNet(2, len(canon.CLASSES), base=c["base"], depth=c["depth"], norm=c.get("norm", "batch"),
               dropout=c.get("dropout", 0.0), skip=c.get("skip", True), coords=c.get("coords", False),
               profile=c.get("profile", False)).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stage", default="E1ext")
    ap.add_argument("--run", default="")
    ap.add_argument("--gate_n", type=int, default=150)
    ap.add_argument("--gate_procs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = a.run or Path(a.weights).parent.name
    rundir = ROOT / "runs" / a.stage / run
    net, c = load_net(ROOT / a.weights, dev)
    cfg = T.TrainConfig(stage=a.stage, run=run, manifest=a.manifest, profile=c.get("profile", False),
                        gate_n=a.gate_n, gate_every=1, gate_procs=a.gate_procs, epochs=1, workers=a.workers)
    df = pd.read_csv(ROOT / a.manifest, low_memory=False).sort_values(["station", "time"]).reset_index(drop=True)
    if a.limit:
        df = df.iloc[np.unique(np.linspace(0, len(df) - 1, a.limit).round().astype(int))].reset_index(drop=True)
    log = tblog.TBLog(rundir, dict(weights=a.weights, manifest=a.manifest, n=len(df), train_cfg=json.dumps(c)))
    t0 = time.time()
    X, Y, P = T.decode(df, a.workers)
    print(f"[{a.stage}/{run}] {len(df)} пар ({df.station.unique().tolist()}, {str(df.time.min())[:10]}…{str(df.time.max())[:10]}) "
          f"декодированы за {time.time() - t0:.0f} с; веса {a.weights}", flush=True)
    vocab = vd.load_vocabulary()
    m, pm, prof, ct = T.evaluate(net, X, Y, P, df, dev, cfg, vocab, 0, log, {})
    m = {k.replace("val/", "ext/"): v for k, v in m.items()}
    # суточные треки по всем суткам
    df["date"] = pd.to_datetime(df.time).dt.strftime("%Y-%m-%d")
    tracks, per_day = [], []
    for date, rows in df.groupby("date"):
        idx = rows.index.values
        d = dict(station=rows.station.iloc[0], date=date, kind="ext")
        t, met = T.day_tracks(net, dev, cfg, d, rows.reset_index(drop=True), X[idx])
        tracks.append(t)
        per_day.append(dict(date=date, n=len(rows), Ap=float(rows.Ap.max()) if "Ap" in rows else np.nan,
                            multi=float(rows.foF1.notna().mean()) if "foF1" in rows else np.nan,
                            **{k.split("/")[-1]: v for k, v in met.items()}))
    days = pd.DataFrame(per_day)
    tr = pd.concat(tracks); tr.to_csv(rundir / "tracks.csv", index=False); days.to_csv(rundir / "days.csv", index=False)
    for key in ("foF2_rmse", "fxI_rmse", "hmF2_rmse", "foF2_corr"):
        if key in days:
            m[f"ext/track_{key}_median"] = float(np.nanmedian(days[key]))
    (rundir / "png").mkdir(exist_ok=True)
    picks = {}
    full = days[days.n >= 48]                                   # только «полные» сутки (≥ 48 ионограмм) — иначе картинка из 1 точки
    if len(full):
        picks["maxAp"] = full.sort_values("Ap", ascending=False).date.iloc[0]
        q = full[full.Ap < 20]
        picks["quiet_multi"] = (q if len(q) else full).sort_values("multi", ascending=False).date.iloc[0]
    for kind, date in picks.items():
        rows = df[df.date == date].reset_index(drop=True); idx = df.index[df.date == date].values
        ctd = T.char_table(*T.predict(net, X[idx], dev, profile=cfg.profile)[::2], rows)
        hours = pd.to_datetime(rows.time).dt.hour + pd.to_datetime(rows.time).dt.minute / 60
        chars = {ch: (ctd[f"{ch}_artist"].values, ctd[f"{ch}_pred"].values, unit) for ch, unit in T.TRACK_CHARS.items()}
        log.tracks_grid(f"track/{rows.station.iloc[0]}_{date}_{kind}", hours.values, chars, 0,
                        title=f"{rows.station.iloc[0]} {date} {kind} (внешний тест)", save=rundir / "png" / f"track_{date}_{kind}.png")
    # панели «как у дигизонда»: 8 образцов равномерно по времени
    ix = np.unique(np.linspace(0, len(df) - 1, min(8, len(df))).round().astype(int))
    titles = [f"{r.station} {str(r.time)[:16]} C{int(r.c_level) if pd.notna(r.c_level) else -1}" for r in df.itertuples()]   # по всей выборке (индексация ix)
    log.digisonde("digisonde/ext", T.digisonde_samples(X, Y, P, pm, prof, df, titles, ix), 0,
                  save=rundir / "png" / "digisonde_ext.png")
    log.scalars(m, 0); log.row(m, 0)
    ct.to_csv(rundir / "val_readouts.csv", index=False)
    json.dump(dict(weights=a.weights, manifest=a.manifest, n=len(df), metrics=m, picks=picks),
              open(rundir / "metrics.json", "w"), indent=1, ensure_ascii=False)
    print(f"  IoU_F2 {m.get('ext/IoU_F2', np.nan):.3f} foF2 RMSE {m.get('ext/foF2_rmse', np.nan):.2f} med {m.get('ext/foF2_med', np.nan):.3f} "
          f"hmF2 RMSE {m.get('ext/hmF2_rmse', np.nan):.1f} prof {m.get('ext/prof_rmse', np.nan):.3f} "
          f"gate {100 * m.get('gate/violations', np.nan):.1f}% (ARTIST {100 * m.get('gate/artist_violations', np.nan):.1f}%) | "
          f"суток {len(days)}, медиана foF2 RMSE по суткам {m.get('ext/track_foF2_rmse_median', np.nan):.2f}", flush=True)
    print(f"[{a.stage}/{run}] готово за {time.time() - t0:.0f} с → {rundir}", flush=True)


if __name__ == "__main__":
    main()
