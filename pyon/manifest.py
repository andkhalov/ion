# -*- coding: utf-8 -*-
"""
manifest.py — опись корпуса: единственная «сборка» вместо тензорных кэшей (Э3 §2.1).

По одной строке на пару (RSF|SBF, SAO): пути, станция, модель зонда, время (из имени
файла), характеристики ARTIST (foF2, hF, foE, foEs, fxI, M3000F2), C-level, флаг
возмущённости (описательные буквы F/Q в SAO), сплит train/val (хронологический,
75/25 внутри станции). Строится за минуты, весит мегабайты; после каждой докачки
корпуса пересобирать.

Запуск:  python -m pyon.manifest [--procs 8] [--limit N]   (--limit -> data/manifest_smoke.csv)
Выход:   data/manifest.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import digi_formats as dfm           # noqa: E402

SCALED_KEYS = ["foF2", "foF1", "hF", "foE", "foEs", "fxI", "M3000F2", "fmin"]
_STEM_RE = re.compile(r"([A-Z0-9]{5})_(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")
# координаты станций корпуса (широта, долгота E) — для местного времени/день-ночь (Э3 §3.6)
STATIONS = {"JI91J": (-12.0, 283.2), "RO041": (41.8, 12.5), "MO155": (55.5, 37.3), "JR055": (54.6, 13.4),
            "PQ052": (50.0, 14.6), "EI764": (64.7, 212.9), "GA762": (62.4, 214.8), "TR169": (69.6, 19.2),
            "NO369": (69.4, 88.1), "YA462": (62.0, 129.6), "SMJ67": (67.0, 309.4)}
RAW_PATTERNS = ["data/corpus/*/*/*/ionogram/*.RSF", "data/corpus/*/*/*/ionogram/*.SBF",
                "data/RSF-samples-w-img-n-sao-n-dft/ionogram/*.RSF",
                "data/SBF-samples-w-img-n-sao/ionogram/*.SBF"]


def collect_files() -> list[str]:
    """Все сырые файлы корпуса и образцов (отсортированный список абсолютных путей)."""
    return sorted({f for pat in RAW_PATTERNS for f in glob.glob(str(ROOT / pat))})


def local_hour(station: str, t) -> float:
    """Местное солнечное время (ч) по долготе станции; NaN для неизвестной станции."""
    if station not in STATIONS:
        return float("nan")
    t = pd.Timestamp(t)
    return (t.hour + t.minute / 60 + STATIONS[station][1] / 15.0) % 24


def daynight(station: str, t) -> str:
    """'day' (6 ≤ LT < 18) | 'night' | 'unknown' — стратификация Э3 §3.6."""
    lt = local_hour(station, t)
    if not np.isfinite(lt):
        return "unknown"
    return "day" if 6 <= lt < 18 else "night"


def stem_time(stem: str):
    """SSSSS_YYYYDDDHHMMSS → (станция, pd.Timestamp) или None."""
    m = _STEM_RE.match(os.path.basename(stem))
    if not m:
        return None
    st, yy, ddd, hh, mm, ss = m.groups()
    return st, pd.Timestamp(int(yy), 1, 1) + pd.Timedelta(days=int(ddd) - 1, hours=int(hh),
                                                          minutes=int(mm), seconds=int(ss))


def _row(f: str):
    try:
        stem = os.path.basename(f)
        parsed = stem_time(stem)
        if parsed is None:
            return None
        sao_f = f.replace("/ionogram/", "/scaled/").rsplit(".", 1)[0] + ".SAO"
        if not os.path.exists(sao_f):
            return None
        st, t = parsed
        row = dict(path=os.path.relpath(f, ROOT), sao=os.path.relpath(sao_f, ROOT),
                   station=st, time=t.isoformat(), fmt=stem.rsplit(".", 1)[-1].upper())
        sao = dfm.read_sao(sao_f)
        sc = sao.get("scaled")
        for k in SCALED_KEYS:
            v = float(sc.get(k, np.nan)) if sc is not None else np.nan
            row[k] = round(v, 3) if np.isfinite(v) else np.nan
        af = sao.get("analysis_flags")
        row["c_level"] = int(af[9]) if af is not None and len(af) >= 10 else -1
        row["model"] = str(sao.get("system_desc", ""))[:6].strip()          # 'DPS-4 '/'DPS-4D'
        letters = "".join(sao.get("desc_letters", []) or [])
        row["disturbed"] = int(any(c in letters for c in "FQ"))             # spread-F метки
        return row
    except Exception:
        return None


def build_manifest(procs: int = 8, limit: int = 0) -> pd.DataFrame:
    files = collect_files()
    if limit:
        files = files[:limit]
    print(f"файлов: {len(files)}; процессов: {procs}", flush=True)
    t0 = time.time()
    with Pool(procs) as pool:
        rows = [r for r in pool.imap_unordered(_row, files, chunksize=64) if r is not None]
    df = pd.DataFrame(rows).sort_values(["station", "time"]).reset_index(drop=True)
    # хронологический сплит 75/25 внутри станции (никаких перемешиваний по времени)
    df["split"] = "train"
    for st, g in df.groupby("station"):
        cut = g.index[int(len(g) * 0.75):]
        df.loc[cut, "split"] = "val"
    print(f"манифест: {len(df)} пар за {time.time()-t0:.0f} с; "
          f"train {(df.split=='train').sum()} / val {(df.split=='val').sum()}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    df = build_manifest(args.procs, args.limit)
    out = ROOT / ("data/manifest_smoke.csv" if args.limit else "data/manifest.csv")
    df.to_csv(out, index=False)
    print("->", out, flush=True)


if __name__ == "__main__":
    main()
