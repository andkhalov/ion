# -*- coding: utf-8 -*-
"""
training.py — тренировочный цикл ВЗ-модели (этап E1: Э3 §2 таблица, §3 метрики, §3.5 наборы
логирования, §3.6 стратификация, §4 протокол; CLAUDE.md §2.3 dry → smoke → full, §2.4 логи).

Модель — «модельное представление ARTIST» (CLAUDE §0): U-Net даёт маску классов (L2) и, через
голову профиля, плазменную частоту fp(h) нижней стороны (NHPC-профиль, L4); характеристики
«как у дигизонда» снимает детерминированный измеритель `pyon.scaler` (foF2, foF1, foE, foEs, fmin,
fxI по fo²=fx(fx−fB), h′F/h′F2/h′E/h′Es, hmF2/hmF1/hmE из профиля, MUF(D)/M(3000) секансом).

Раны E1 (VARIANTS): baseline (CE) | +hinge (CE + λ·L_logic hinge) | +lognorm (softplus) |
+coords (координатные каналы, абляция). Данные — потоковый лоадер по манифесту (Э3 §2.2):
train — DataLoader с воркерами (нулевая эпоха — дискодружественный блочный порядок), далее
ОГРАНИЧЕННЫЙ RAM-кэш декодированных образцов (§2.2, IO-предел диска); фиксированный
val-поднабор декодируется один раз в RAM.

Каждая эпоха: обучение (AMP; лосс = CE + λ·L_logic + λp·(L1 профиля + BCE валидности)) → оценка:
CE; hinge-мера L_logic покомпонентно + гистограмма по образцам; IoU по классам, precision/recall
следа; таблица характеристик против ARTIST (RMSE, медиана, bias; для foF2 — доли «точно по РД» /
«сомнительно», РД 52.26.817 §6.5.5); невязка профиля (RMSE fp по высотам ARTIST-валидности, IoU
валидности, |ΔhmF2|); стратификация (станция, день/ночь, C-level ≤22/>22, спокойные/возмущённые);
SHACL-гейт на gate_n сценах с гирочастотой станции (+ ARTIST-референс); панели «как у дигизонда»
на фиксированном наборе показательных ионограмм (эхо, следы ARTIST, наши контуры, профили
ARTIST vs наш, таблица характеристик); суточные треки нескольких характеристик с разбросом на
фиксированных сутках (RMSE, bias, корреляция хода, Э3 §3.5).

Артефакты runs/<этап>/<ран>/: events (TensorBoard), metrics.csv (строка на эпоху, все скаляры),
weights.pt (ЛУЧШАЯ эпоха по best_metric) и weights_last.pt (последняя; state_dict + cfg + метрики),
config.json (конфиг + provenance: git-коммит, md5 манифеста, версии, GPU, argv, время старта),
val_readouts.csv (характеристики лучшей модели по val-поднабору против ARTIST + невязка профиля),
tracks.csv (суточные треки лучшей модели), fixed_set_preds.npz (вход, цель, предсказание, профили
фиксированного набора), summary.json (best и last эпохи). Набор логирования —
runs/<этап>/logging_set.json (создаётся ОДИН раз и не меняется, мотивация — внутри файла и в DIARY).

Запуск:  python -m pyon.training --stage E1 --run baseline --variant baseline --manifest data/manifest.csv
         python -m pyon.training --dry            (1 шаг, 8 val, 2 сцены; проверка форм и путей)
         python -m pyon.training --stage smoke --run hinge --variant hinge --manifest data/manifest_smoke.csv \\
                --epochs 1 --gate_n 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import canon, gates, loader, logic, manifest, scaler, tblog   # noqa: E402
from pyon import digi_formats as dfm                                       # noqa: E402
from pyon import validate as vd                                            # noqa: E402
from pyon.models import UNet, n_params                                     # noqa: E402

VARIANTS = {"baseline": dict(logic=None, coords=False), "hinge": dict(logic="hinge", coords=False),
            "lognorm": dict(logic="lognorm", coords=False), "coords": dict(logic=None, coords=True)}
CE_WEIGHTS = [0.05, 1.0, 1.0, 1.0, 1.0]
CHAR_STATS = ["foF2", "foF1", "foE", "foEs", "fmin", "fxI", "hF", "hF2", "hE", "hEs", "hmF2", "MUF3000"]
TRACK_CHARS = {"foF2": "МГц", "fxI": "МГц", "fmin": "МГц", "hF": "км", "hmF2": "км"}


@dataclass
class TrainConfig:
    run: str = "baseline"
    stage: str = "E1"
    manifest: str = "data/manifest.csv"
    variant: str = "baseline"       # baseline | hinge | lognorm | coords
    lam: float = 0.5                # вес L_logic
    profile: bool = True            # голова профиля fp(h) (NHPC-цель)
    lam_prof: float = 0.5           # вес лосса профиля (L1 в МГц + BCE валидности)
    epochs: int = 10
    batch: int = 64
    lr: float = 2e-3
    sched: str = "cosine"           # const | cosine (CosineAnnealingLR по эпохам до lr/100)
    amp: bool = True
    workers: int = 8
    seed: int = 0
    limit: int = 0                  # подмножество: по половине лимита из каждого сплита, равномерно
    val_size: int = 4000            # фиксированный val-поднабор (в RAM), поровну по станциям
    gate_n: int = 150               # сцен на SHACL-гейт
    gate_every: int = 5             # гейт каждые N эпох и в последнюю
    gate_procs: int = 4
    log_images: int = 48            # максимум панелей фиксированного набора
    images_every: int = 2
    max_steps: int = 0              # dry: шагов в эпохе
    holdout: str = ""               # leave-one-station-out: станция исключается из train
    block_shuffle: int = 2048       # блочный порядок чтения в нулевой эпохе (0 = случайный)
    cache_gb: float = 10.0          # RAM-кэш train (Э3 §2.2): 0 = только поток; p1 (~180 тыс.) ≈ 9 ГБ
    # архитектура (Э3 §7: d4 b32 BN, без dropout, со skip)
    depth: int = 4
    base: int = 32
    norm: str = "batch"
    dropout: float = 0.0
    skip: bool = True
    best_metric: str = "val/foF2_med"
    dry: bool = False
    device: str = "cuda"
    render_train: str = ""          # сим2реал (Э3 §3.4): веса рендерера → входы ОБУЧЕНИЯ рендерятся из масок каждый батч
    render_val: str = ""            # веса рендерера → val/панели/треки оцениваются на РЕНДЕРЕ (клетки «→рендер»)
    bg_mode: str = ""               # трансплантация реального фона при render_train: "own" (фон того же образца) |
                                    # "shuffle" (фон другого образца батча); "" — чистый рендер (Э3 §2 E4)
    bg_dilate: int = 2              # радиус расширения маски следов (px) — внутри рендер, снаружи реальный фон


# ---------------------------------------------------------------------------- служебное
def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "?"
    except Exception:
        return "?"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------- данные
def load_split(cfg: TrainConfig):
    df = pd.read_csv(ROOT / cfg.manifest)
    if cfg.limit:
        parts = []
        for _, g in df.groupby("split"):
            n = min(len(g), max(1, cfg.limit // 2))
            parts.append(g.iloc[np.unique(np.linspace(0, len(g) - 1, n).round().astype(int))])
        df = pd.concat(parts)
    df = df.reset_index(drop=True)
    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    if cfg.holdout:
        tr = tr[tr.station != cfg.holdout]
    n_st = max(1, va.station.nunique())
    per = max(1, cfg.val_size // n_st)
    parts = []
    for _, g in va.groupby("station"):
        idx = np.unique(np.linspace(0, len(g) - 1, min(per, len(g))).round().astype(int))
        parts.append(g.iloc[idx])
    va_sub = pd.concat(parts).sort_values(["station", "time"]).reset_index(drop=True)
    return df, tr.reset_index(drop=True), va_sub


def decode(df: pd.DataFrame, workers: int, batch: int = 128):
    """Декодировать все пары df в RAM: X uint8 [N,2,H,W], Y int8 [N,H,W], P float32 [N,H]."""
    if not len(df):
        return (torch.zeros((0, 2, canon.NH, canon.NF), dtype=torch.uint8),
                torch.zeros((0, canon.NH, canon.NF), dtype=torch.int8), torch.zeros((0, canon.NH)))
    dl = DataLoader(loader.VerticalDataset(df), batch_size=batch, shuffle=False, num_workers=workers)
    xs, ys, ps = zip(*[(x, y, p) for x, y, p in dl])
    return torch.cat(xs), torch.cat(ys), torch.cat(ps)


def to_input(x: torch.Tensor, dev) -> torch.Tensor:
    return x.to(dev, non_blocking=True).float().div_(255)


@torch.no_grad()
def transplant(x_real: torch.Tensor, x_render: torch.Tensor, y: torch.Tensor, mode: str, dilate: int = 2) -> torch.Tensor:
    """Трансплантация реального фона (Э3 §2 E4): внутри маски следов, расширенной на dilate px, — рендер;
    снаружи — реальное сырьё того же образца ("own") или другого образца батча ("shuffle")."""
    k = 2 * dilate + 1
    region = torch.nn.functional.max_pool2d((y > 0).float().unsqueeze(1), k, stride=1, padding=dilate) > 0
    bg = x_real if mode == "own" else x_real[torch.randperm(len(x_real), device=x_real.device)]
    return torch.where(region, x_render, bg)


@torch.no_grad()
def render_uint8(ren, Y: torch.Tensor, dev, seed: int = 0, batch: int = 128) -> torch.Tensor:
    """Маски [N,H,W] → рендер [N,2,H,W] uint8 0..255 (как сырьё лоадера); детерминизм — по seed."""
    torch.manual_seed(seed)
    out = [(ren.sample(Y[k:k + batch].to(dev).long()) * 255).round().to(torch.uint8).cpu() for k in range(0, len(Y), batch)]
    return torch.cat(out) if out else torch.zeros((0, 2, canon.NH, canon.NF), dtype=torch.uint8)


def gyros_of(df: pd.DataFrame) -> np.ndarray:
    g = df["gyro"].astype(float).values if "gyro" in df else np.full(len(df), np.nan)
    return np.where(np.isfinite(g), g, 1.3)


def select_logging_set(va: pd.DataFrame, path: Path, per_cell: int = 2, max_scan: int = 2000) -> dict:
    """Э3 §3.5 + «показательные примеры» (решение АХ 2026-09-05): на станцию по per_cell ионограмм
    в категориях: day_rich — день, спокойные, ≥3 классов в разметке ARTIST (E+F1+F2 [+Es]) с лучшим
    C-level; night — ночь, спокойные, лучший C-level (кратники ночью сильны); es — есть Es;
    disturbed — возмущённые сутки (Ap ≥ 30 ∨ Kp ≥ 5 ∨ буквы F/Q); weak_artist — C-level ≥ 44
    (ARTIST слаб — там ценен онтологический контур). Сутки для треков: quiet — спокойные сутки с
    максимальной долей многослойных ионограмм (≥ 48 ионограмм); disturbed — сутки с max Ap
    (≥ 24). Создаётся ОДИН раз (файл path); мотивация — в самом файле и в DIARY."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    va = va.copy()
    va["dn"] = [manifest.daynight(s, t) for s, t in zip(va.station, va.time)]
    va["date"] = pd.to_datetime(va.time).dt.date.astype(str)
    images, days = [], []
    for st, g in va.groupby("station"):
        scan = g.iloc[np.unique(np.linspace(0, len(g) - 1, min(max_scan, len(g))).round().astype(int))].copy()
        ncls, has_es = [], []
        for r in scan.itertuples():
            try:
                y = canon.masks_from_sao(dfm.read_sao(str(ROOT / r.sao)))
                u = set(np.unique(y).tolist()) - {0}
            except Exception:
                u = set()
            ncls.append(len(u)); has_es.append(canon.CLASSES.index("Es") in u)
        scan["ncls"], scan["has_es"] = ncls, has_es
        quiet = scan[scan.disturbed == 0]
        cells = {
            "day_rich": quiet[(quiet.dn == "day") & (quiet.ncls >= 3)].sort_values(["ncls", "c_level"], ascending=[False, True]),
            "night": quiet[quiet.dn == "night"].sort_values("c_level"),
            "es": quiet[quiet.has_es].sort_values("c_level"),
            "disturbed": scan[scan.disturbed == 1].sort_values("c_level"),
            "weak_artist": scan[scan.c_level >= 44].sort_values("c_level", ascending=False),
        }
        used = set()
        for cat, cand in cells.items():
            for r in cand.itertuples():
                if len([i for i in images if i["station"] == st and i["cat"] == cat]) >= per_cell:
                    break
                if r.path in used:
                    continue
                used.add(r.path)
                images.append(dict(path=r.path, station=st, time=r.time, dn=r.dn, disturbed=int(r.disturbed),
                                   c_level=int(r.c_level), ncls=int(r.ncls), cat=cat))
        by_day = scan.groupby("date").agg(n=("path", "size"), dist=("disturbed", "sum"), rich=("ncls", "mean"),
                                          Ap=("Ap", "max"))
        full = g.groupby("date").size()
        by_day["n_full"] = full.reindex(by_day.index).values
        q = by_day[(by_day.dist == 0) & (by_day.n_full >= 48)]
        if len(q):
            days.append(dict(station=st, date=q.rich.idxmax(), kind="quiet", n=int(q.n_full[q.rich.idxmax()])))
        dd = by_day[(by_day.dist > 0) & (by_day.n_full >= 24)]
        if len(dd):
            days.append(dict(station=st, date=dd.Ap.idxmax(), kind="disturbed", n=int(dd.n_full[dd.Ap.idxmax()]),
                             Ap=float(dd.Ap.max())))
    out = dict(images=images, days=days, created=time.strftime("%Y-%m-%d %H:%M"), manifest_rows=int(len(va)),
               motivation="категории: day_rich (E+F1+F2, лучший C-level), night (кратники), es, disturbed "
                          "(Ap≥30∨Kp≥5∨F/Q), weak_artist (C-level≥44); сутки: quiet = max доля многослойных, "
                          "disturbed = max Ap. Набор фиксирован на этап и не меняется между ранами.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------- метрики
def err_stats(pred, true, fo_true=None) -> dict:
    """RMSE, медиана |Δ|, bias, n; для foF2 — доли «точно по РД» (0.1 МГц | 2 %) и «сомнительно» (0.2 | 5 %)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(pred) & np.isfinite(true)
    out = dict(n=int(ok.sum()), n_pred=int(np.isfinite(pred).sum()), n_true=int(np.isfinite(true).sum()))
    if not ok.any():
        return out
    d = pred[ok] - true[ok]
    out.update(rmse=float(np.sqrt(np.mean(d ** 2))), med=float(np.median(np.abs(d))), bias=float(np.mean(d)))
    if fo_true is not None:
        t = np.asarray(fo_true, float)[ok]
        out["rd_exact"] = float(np.mean(np.abs(d) <= np.maximum(0.1, 0.02 * t)))
        out["rd_doubt"] = float(np.mean(np.abs(d) <= np.maximum(0.2, 0.05 * t)))
    return out


def strata(meta: pd.DataFrame) -> dict:
    dn = np.array([manifest.daynight(s, t) for s, t in zip(meta.station, meta.time)])
    out = {f"station_{st}": (meta.station.values == st) for st in sorted(meta.station.unique())}
    out.update({"day": dn == "day", "night": dn == "night",
                "clevel_le22": meta.c_level.values <= 22, "clevel_gt22": meta.c_level.values > 22,
                "quiet": meta.disturbed.values == 0, "disturbed": meta.disturbed.values == 1})
    return {k: v for k, v in out.items() if v.sum() > 0}


@torch.no_grad()
def predict(net, X: torch.Tensor, dev, batch: int = 128, profile: bool = False):
    """→ argmax-маски [N,H,W] int8, hinge-L_logic по компонентам [N], профиль [N,3,H] (fp, p_valid, hmF2-доля) или None."""
    net.eval()
    pms, comps, profs = [], {}, []
    if not len(X):
        return np.zeros((0, canon.NH, canon.NF), np.int8), {}, (np.zeros((0, 3, canon.NH), np.float32) if profile else None)
    for k in range(0, len(X), batch):
        xb = to_input(X[k:k + batch], dev)
        if profile:
            lg, pr = net(xb, profile=True)
            profs.append(torch.stack([pr[:, 0], torch.sigmoid(pr[:, 1]), pr[:, 2]], 1).float().cpu())
        else:
            lg = net(xb)
        lg = lg.float()
        pms.append(lg.argmax(1).to(torch.int8).cpu())
        _, c = logic.vertical_logic(lg, "hinge")
        for name, v in c.items():
            comps.setdefault(name, []).append(v.detach().cpu())
    pm = torch.cat(pms).numpy()
    comps = {k: torch.cat(v).numpy() for k, v in comps.items()}
    return pm, comps, (torch.cat(profs).numpy() if profile else None)


def hm_of(prof: np.ndarray | None, k: int) -> float:
    """hmF2 (км) из канала прямой регрессии головы профиля; NaN, если профиля нет."""
    if prof is None or prof.shape[1] < 3:
        return float("nan")
    return float(canon.H_MIN + prof[k, 2, 0] * (canon.H_MAX - canon.H_MIN))


def masked_profile(prof: np.ndarray | None, k: int):
    """fp(h) предсказания с NaN там, где p_valid < 0.5."""
    if prof is None:
        return None
    fp, pv = prof[k, 0].astype(float), prof[k, 1]
    fp = fp.copy(); fp[pv < 0.5] = np.nan
    return fp


@torch.no_grad()
def val_ce(net, X, Y, dev, batch: int = 128) -> float:
    net.eval()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(CE_WEIGHTS, device=dev))
    tot = n = 0
    for k in range(0, len(X), batch):
        lg = net(to_input(X[k:k + batch], dev)).float()
        tot += ce(lg, Y[k:k + batch].to(dev).long()).item() * len(lg); n += len(lg)
    return tot / max(n, 1)


def segmentation_metrics(pm: np.ndarray, Y: np.ndarray) -> dict:
    out = {}
    for ci, cls in enumerate(canon.CLASSES[1:], 1):
        inter = ((pm == ci) & (Y == ci)).sum(); union = ((pm == ci) | (Y == ci)).sum()
        out[f"IoU_{cls}"] = float(inter / union) if union else np.nan
    p, t = pm > 0, Y > 0
    tp = (p & t).sum()
    out["trace_precision"] = float(tp / p.sum()) if p.sum() else np.nan
    out["trace_recall"] = float(tp / t.sum()) if t.sum() else np.nan
    return out


def char_table(pm: np.ndarray, prof: np.ndarray | None, meta: pd.DataFrame) -> pd.DataFrame:
    """Характеристики измерителя по каждому образцу (+ ARTIST из манифеста)."""
    gy = gyros_of(meta)
    rows = []
    for k in range(len(pm)):
        r = scaler.scale_vertical(pm[k], masked_profile(prof, k), f_b=float(gy[k]), hm_f2=hm_of(prof, k))
        out = {f"{n}_pred": r.get(n, np.nan) for n in CHAR_STATS + ["M3000F2", "hmF1", "hmE"]}
        for n in CHAR_STATS + ["M3000F2"]:
            key = scaler.ARTIST_KEYS.get(n, n)
            out[f"{n}_artist"] = float(meta[key].iloc[k]) if key in meta else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def profile_metrics(prof: np.ndarray, P: np.ndarray) -> tuple[dict, np.ndarray]:
    """Невязка профилей: RMSE fp по ARTIST-валидным высотам (на образец), IoU валидности."""
    fp, pv = prof[:, 0], prof[:, 1] >= 0.5
    okt = np.isfinite(P)
    d = np.where(okt, fp - np.nan_to_num(P), 0.0)
    n = okt.sum(1)
    rmse = np.sqrt(np.where(n > 0, (d ** 2).sum(1) / np.maximum(n, 1), np.nan))
    inter = (pv & okt).sum(1); union = (pv | okt).sum(1)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return dict(prof_rmse=float(np.nanmean(rmse)), prof_rmse_med=float(np.nanmedian(rmse)),
                prof_valid_iou=float(np.nanmean(iou)), prof_n=int(np.isfinite(rmse).sum())), rmse


def evaluate(net, X, Y, P, meta, dev, cfg: TrainConfig, vocab, epoch: int, log: tblog.TBLog, artist_gate: dict):
    m = {"val/CE": val_ce(net, X, Y, dev)}
    pm, comps, prof = predict(net, X, dev, profile=cfg.profile)
    Yn = Y.numpy()
    for k, v in comps.items():
        m[f"val/logic_{k}"] = float(v.mean())
    tot = sum(comps.values())
    m["val/logic_total"] = float(tot.mean())
    log.hist("val/logic_total_hist", tot, epoch)
    m.update({f"val/{k}": v for k, v in segmentation_metrics(pm, Yn).items()})
    ct = char_table(pm, prof, meta)
    for name in CHAR_STATS:
        st = err_stats(ct[f"{name}_pred"], ct[f"{name}_artist"], ct["foF2_artist"] if name == "foF2" else None)
        for k, v in st.items():
            m[f"val/{name}_{k}"] = v
    prof_rmse = None
    if cfg.profile:
        pmet, prof_rmse = profile_metrics(prof, P.numpy())
        m.update({f"val/{k}": v for k, v in pmet.items()})
        log.hist("val/prof_rmse_hist", prof_rmse, epoch)
        ct["prof_rmse"] = prof_rmse
    for sname, mask in strata(meta).items():
        if mask.sum() < 5:
            continue
        st = err_stats(ct.foF2_pred[mask], ct.foF2_artist[mask])
        m[f"strat/{sname}/foF2_rmse"] = st.get("rmse", np.nan)
        m[f"strat/{sname}/foF2_med"] = st.get("med", np.nan)
        m[f"strat/{sname}/IoU_F2"] = segmentation_metrics(pm[mask], Yn[mask])["IoU_F2"]
        if prof_rmse is not None:
            m[f"strat/{sname}/prof_rmse"] = float(np.nanmean(prof_rmse[mask]))
        m[f"strat/{sname}/n"] = int(mask.sum())
    n_g = min(cfg.gate_n, len(pm))
    if n_g and (epoch % cfg.gate_every == 0 or epoch == cfg.epochs - 1):
        t_g = time.time()
        gy = gyros_of(meta)[:n_g]
        rate, warn, _ = gates.gate_rate(pm[:n_g], gates.vertical_scene, vocab, prefix=f"e{epoch}_",
                                        procs=cfg.gate_procs, with_warnings=True, gyros=gy)
        m["gate/violations"], m["gate/warnings"] = rate, warn
        if "artist" not in artist_gate:
            artist_gate["artist"], artist_gate["artist_w"], _ = gates.gate_rate(
                Yn[:n_g], gates.vertical_scene, vocab, prefix="art_", procs=cfg.gate_procs, with_warnings=True, gyros=gy)
        m["gate/artist_violations"], m["gate/artist_warnings"] = artist_gate["artist"], artist_gate["artist_w"]
        m["gate/n"] = n_g
        m["time/gate_s"] = time.time() - t_g
    ct.insert(0, "logic_total", tot)
    return m, pm, prof, ct


def digisonde_samples(X, Y, P, pm, prof, rows: pd.DataFrame, titles, idx) -> list:
    """Список образцов для tblog.digisonde по индексам idx."""
    gy = gyros_of(rows)
    out = []
    for i in idx:
        pp = masked_profile(prof, i) if prof is not None else np.full(canon.NH, np.nan)
        ours = scaler.scale_vertical(pm[i], pp, f_b=float(gy[i]), hm_f2=hm_of(prof, i))
        table = []
        for name in scaler.REPORT_ROWS:
            key = scaler.ARTIST_KEYS.get(name, name)
            a = float(rows[key].iloc[i]) if key in rows else np.nan
            table.append((name, a, float(ours.get(name, np.nan))))
        out.append(dict(x=X[i].numpy(), y=Y[i].numpy(), pm=pm[i], p_true=P[i].numpy(),
                        p_pred=prof[i, 0] if prof is not None else np.full(canon.NH, np.nan),
                        p_valid=(prof[i, 1] >= 0.5) if prof is not None else np.zeros(canon.NH, bool),
                        table=table, title=titles[i]))
    return out


def day_tracks(net, dev, cfg, d, rows, Xd) -> tuple[pd.DataFrame, dict]:
    """Треки характеристик за фиксированные сутки: DataFrame (time, <char>_artist, <char>_pred) + метрики."""
    pmd, _, profd = predict(net, Xd, dev, profile=cfg.profile)
    ct = char_table(pmd, profd, rows)
    t = pd.DataFrame({"station": d["station"], "date": d["date"], "kind": d["kind"], "time": rows.time.values})
    met = {}
    key = f"{d['station']}_{d['date']}_{d['kind']}"
    for c in TRACK_CHARS:
        t[f"{c}_artist"], t[f"{c}_pred"] = ct[f"{c}_artist"].values, ct[f"{c}_pred"].values
        st = err_stats(ct[f"{c}_pred"], ct[f"{c}_artist"])
        met[f"track/{key}/{c}_rmse"] = st.get("rmse", np.nan)
        a, b = ct[f"{c}_artist"].values.astype(float), ct[f"{c}_pred"].values.astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        met[f"track/{key}/{c}_corr"] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan
    return t, met


# ---------------------------------------------------------------------------- цикл
def train(cfg: TrainConfig) -> dict:
    if cfg.dry:
        cfg.limit = cfg.limit or 48; cfg.epochs = 1; cfg.max_steps = 1; cfg.val_size = 8
        cfg.gate_n = 2; cfg.log_images = 2; cfg.workers = min(cfg.workers, 2); cfg.gate_procs = 2
        cfg.stage = "dry"
    set_seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    v = VARIANTS[cfg.variant]
    rundir = ROOT / "runs" / cfg.stage / cfg.run
    prov = dict(device=str(dev), torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), python=sys.version.split()[0],
                gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "-",
                git_commit=_git_commit(), manifest_md5=_md5(ROOT / cfg.manifest), argv=" ".join(sys.argv),
                started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    log = tblog.TBLog(rundir, asdict(cfg) | prov)
    t_all = time.time()

    df, tr, va = load_split(cfg)
    print(f"[{cfg.stage}/{cfg.run}] манифест {cfg.manifest}: {len(df)} строк; train {len(tr)}, val-поднабор {len(va)} "
          f"({va.station.nunique()} станций); устройство {dev}", flush=True)
    sampler = loader.BlockShuffleSampler(len(tr), cfg.block_shuffle, seed=cfg.seed) if cfg.block_shuffle else None
    dl = DataLoader(loader.WithIndex(loader.VerticalDataset(tr)), batch_size=cfg.batch, shuffle=sampler is None,
                    sampler=sampler, num_workers=cfg.workers, persistent_workers=cfg.workers > 0,
                    prefetch_factor=4 if cfg.workers > 0 else None, drop_last=False, pin_memory=dev.type == "cuda")
    per_sample = 2 * canon.NH * canon.NF + canon.NH * canon.NF + 4 * canon.NH
    use_cache = cfg.cache_gb > 0 and len(tr) * per_sample <= cfg.cache_gb * 2 ** 30
    if use_cache:
        Xc = torch.empty((len(tr), 2, canon.NH, canon.NF), dtype=torch.uint8)
        Yc = torch.empty((len(tr), canon.NH, canon.NF), dtype=torch.int8)
        Pc = torch.empty((len(tr), canon.NH), dtype=torch.float32)
        cached = torch.zeros(len(tr), dtype=torch.bool)
    print(f"  RAM-кэш train: {'ВКЛ' if use_cache else 'выкл'} ({len(tr) * per_sample / 2**30:.1f} ГБ нужно, лимит {cfg.cache_gb} ГБ)", flush=True)
    t0 = time.time()
    Xv, Yv, Pv = decode(va, cfg.workers)
    from pyon import renderer as rnd                                  # ленивый импорт: renderer импортирует training
    ren_t = rnd.load_renderer(ROOT / cfg.render_train, dev) if cfg.render_train else None
    ren_v = rnd.load_renderer(ROOT / cfg.render_val, dev) if cfg.render_val else None
    if ren_v is not None:
        Xv = render_uint8(ren_v, Yv, dev, seed=cfg.seed + 5)
        print(f"  сим2реал: val-входы отрендерены ({cfg.render_val})", flush=True)
    print(f"  val-поднабор: X {tuple(Xv.shape)}, Y {tuple(Yv.shape)}, P {tuple(Pv.shape)} "
          f"(профиль есть у {int(torch.isfinite(Pv).any(1).sum())}), {time.time() - t0:.1f} с", flush=True)
    lset = select_logging_set(df[df.split == "val"], ROOT / "runs" / cfg.stage / "logging_set.json")
    img_rows = df[df.path.isin([im["path"] for im in lset["images"]])].sort_values(["station", "time"]).head(cfg.log_images)
    img_rows = img_rows.reset_index(drop=True)
    Xi, Yi, Pi = decode(img_rows, min(cfg.workers, 2))
    if ren_v is not None and len(Yi):
        Xi = render_uint8(ren_v, Yi, dev, seed=cfg.seed + 6)
    cat_of = {im["path"]: im.get("cat", "") for im in lset["images"]}
    img_titles = [f"{r.station} {str(r.time)[:16]} {manifest.daynight(r.station, r.time)} C{int(r.c_level)} "
                  f"{cat_of.get(r.path, '')}" + (" ВОЗМ" if r.disturbed else "") for r in img_rows.itertuples()]
    img_groups = {st: np.flatnonzero(img_rows.station.values == st) for st in img_rows.station.unique()}
    day_sets = []
    for d in lset["days"]:
        rows = df[(df.station == d["station"]) & (pd.to_datetime(df.time).dt.date.astype(str) == d["date"])].sort_values("time")
        Xd, Yd_, _ = decode(rows, min(cfg.workers, 2))
        if ren_v is not None and len(Yd_):
            Xd = render_uint8(ren_v, Yd_, dev, seed=cfg.seed + 7)
        day_sets.append((d, rows.reset_index(drop=True), Xd))
    print(f"  набор логирования: {len(img_rows)} панелей ({', '.join(f'{k}:{len(v)}' for k, v in img_groups.items())}), "
          f"{len(day_sets)} суток-треков", flush=True)

    net = UNet(2, len(canon.CLASSES), base=cfg.base, depth=cfg.depth, norm=cfg.norm, dropout=cfg.dropout,
               skip=cfg.skip, coords=v["coords"], profile=cfg.profile).to(dev)
    opt = torch.optim.Adam(net.parameters(), cfg.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs), eta_min=cfg.lr / 100)
             if cfg.sched == "cosine" else None)
    scaler_amp = torch.amp.GradScaler(enabled=(cfg.amp and dev.type == "cuda"))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(CE_WEIGHTS, device=dev))
    bce = nn.BCEWithLogitsLoss()
    h_frac = torch.linspace(0, 1, canon.NH, device=dev)           # доля решётки высот для регрессии hmF2
    H_SPAN100 = (canon.H_MAX - canon.H_MIN) / 100.0
    vocab = vd.load_vocabulary()
    artist_gate: dict = {}
    best = dict(value=float("inf"), epoch=-1)
    print(f"  U-Net {n_params(net)} параметров (depth {cfg.depth}, base {cfg.base}, norm {cfg.norm}, dropout {cfg.dropout}, "
          f"skip {cfg.skip}, profile {cfg.profile}), вариант {cfg.variant} (logic={v['logic']}, coords={v['coords']}), "
          f"AMP={scaler_amp.is_enabled()}, sched={cfg.sched}", flush=True)

    hist = []
    for ep in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        net.train(); t_ep = time.time(); n_seen = 0
        sums = {"CE": 0.0, "logic": 0.0, "prof": 0.0}; csum = {}
        from_ram = use_cache and bool(cached.all())
        if from_ram:
            perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(cfg.seed * 1000 + ep))
            batches = ((Xc[j], Yc[j], Pc[j], j) for j in perm.split(cfg.batch))
        else:
            batches = dl
        for step, (x, y, p, idx) in enumerate(batches):
            if cfg.max_steps and step >= cfg.max_steps:
                break
            if use_cache and not from_ram:
                Xc[idx] = x; Yc[idx] = y; Pc[idx] = p; cached[idx] = True
            x, y, p = to_input(x, dev), y.to(dev, non_blocking=True).long(), p.to(dev, non_blocking=True)
            if ren_t is not None:
                with torch.no_grad():
                    x_r = ren_t.sample(y)                             # новая реализация спекла на каждый показ
                    x = transplant(x, x_r, y, cfg.bg_mode, cfg.bg_dilate) if cfg.bg_mode else x_r
            with torch.autocast("cuda", enabled=scaler_amp.is_enabled()):
                if cfg.profile:
                    lg, pr = net(x, profile=True)
                else:
                    lg = net(x)
                loss_ce = ce(lg, y)
            loss = loss_ce
            if v["logic"]:
                l_logic, comps = logic.vertical_logic(lg.float(), v["logic"])
                loss = loss + cfg.lam * l_logic
                sums["logic"] += l_logic.item() * len(x)
                for k, c in comps.items():
                    csum[k] = csum.get(k, 0.0) + c.mean().item() * len(x)
            if cfg.profile:
                pr = pr.float(); valid = torch.isfinite(p)
                l1 = (pr[:, 0] - torch.nan_to_num(p)).abs()[valid].mean() if valid.any() else lg.sum() * 0
                has = valid.any(1)                                            # hmF2 = верх валидного профиля
                hm_t = (valid.float() * h_frac).amax(1)
                l_hm = (pr[:, 2, 0] - hm_t).abs()[has].mean() * H_SPAN100 if has.any() else lg.sum() * 0
                l_prof = l1 + 0.2 * bce(pr[:, 1], valid.float()) + l_hm                # l_hm — в сотнях км
                loss = loss + cfg.lam_prof * l_prof
                sums["prof"] += l_prof.item() * len(x)
            opt.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update()
            sums["CE"] += loss_ce.item() * len(x); n_seen += len(x)
            if cfg.dry:
                print(f"  dry: x {tuple(x.shape)} y {tuple(y.shape)} p {tuple(p.shape)} logits {tuple(lg.shape)} "
                      f"CE {loss_ce.item():.3f} prof {sums['prof'] / len(x):.3f} loss {loss.item():.3f}", flush=True)
        t_train = time.time() - t_ep
        m = {"train/CE": sums["CE"] / max(n_seen, 1), "time/train_s": t_train,
             "time/train_samples_per_s": n_seen / max(t_train, 1e-9), "epoch": ep,
             "train/lr": opt.param_groups[0]["lr"], "train/from_ram": int(from_ram)}
        if v["logic"]:
            m["train/logic"] = sums["logic"] / max(n_seen, 1)
            m.update({f"train/logic_{k}": s_ / max(n_seen, 1) for k, s_ in csum.items()})
        if cfg.profile:
            m["train/prof"] = sums["prof"] / max(n_seen, 1)
        if sched is not None:
            sched.step()
        t_ev = time.time()
        mv, pm, prof, ct = evaluate(net, Xv, Yv, Pv, va, dev, cfg, vocab, ep, log, artist_gate)
        m.update(mv)
        if len(Xi) and (ep % cfg.images_every == 0 or ep == cfg.epochs - 1):
            pmi, _, profi = predict(net, Xi, dev, profile=cfg.profile)
            (rundir / "png").mkdir(exist_ok=True)
            for st, ix in img_groups.items():
                log.digisonde(f"digisonde/{st}", digisonde_samples(Xi, Yi, Pi, pmi, profi, img_rows, img_titles, ix), ep,
                              save=rundir / "png" / f"digisonde_{st}_ep{ep:02d}.png")
        for d, rows, Xd in day_sets:
            if not len(Xd):
                continue
            t_df, met = day_tracks(net, dev, cfg, d, rows, Xd)
            m.update(met)
            key = f"{d['station']}_{d['date']}_{d['kind']}"
            hours = pd.to_datetime(rows.time).dt.hour.values + pd.to_datetime(rows.time).dt.minute.values / 60
            (rundir / "png").mkdir(exist_ok=True)
            log.tracks_grid(f"track/{key}", hours, {c: (t_df[f"{c}_artist"].values, t_df[f"{c}_pred"].values, u)
                                                  for c, u in TRACK_CHARS.items()}, ep, title=key,
                            save=rundir / "png" / f"track_{key}_ep{ep:02d}.png")
        m["time/eval_s"] = time.time() - t_ev
        log.scalars({k: v_ for k, v_ in m.items() if k != "epoch"}, ep)
        log.row(m, ep); hist.append(m)
        ckpt = {"state_dict": net.state_dict(), "cfg": asdict(cfg), "epoch": ep, "metrics": m}
        torch.save(ckpt, rundir / "weights_last.pt")
        crit = m.get(cfg.best_metric, np.nan)
        if np.isfinite(crit) and crit < best["value"]:
            best = dict(value=float(crit), epoch=ep)
            torch.save(ckpt, rundir / "weights.pt")
        print(f"  ep{ep}: train CE {m['train/CE']:.3f}" + (f" logic {m['train/logic']:.4f}" if v["logic"] else "")
              + (f" prof {m['train/prof']:.3f}" if cfg.profile else "")
              + f" | val CE {m['val/CE']:.3f} L_hinge {m['val/logic_total']:.4f} IoU_F2 {m.get('val/IoU_F2', np.nan):.3f} "
              f"foF2 RMSE {m.get('val/foF2_rmse', np.nan):.2f} med {m.get('val/foF2_med', np.nan):.2f} "
              f"hmF2 RMSE {m.get('val/hmF2_rmse', np.nan):.1f} prof RMSE {m.get('val/prof_rmse', np.nan):.2f} "
              f"gate {m.get('gate/violations', np.nan):.0%} (ARTIST {m.get('gate/artist_violations', np.nan):.0%}) "
              f"| {t_train:.0f}+{m['time/eval_s']:.0f} с, {m['time/train_samples_per_s']:.0f} обр/с", flush=True)

    # финальные артефакты — от ЛУЧШИХ весов
    last_m = hist[-1]
    if best["epoch"] >= 0 and best["epoch"] != cfg.epochs - 1:
        net.load_state_dict(torch.load(rundir / "weights.pt", map_location=dev)["state_dict"])
        pm, comps, prof = predict(net, Xv, dev, profile=cfg.profile)
        ct = char_table(pm, prof, va); ct.insert(0, "logic_total", sum(comps.values()))
        if cfg.profile:
            ct["prof_rmse"] = profile_metrics(prof, Pv.numpy())[1]
    ct.insert(0, "time", va.time.values); ct.insert(0, "station", va.station.values); ct.insert(0, "path", va.path.values)
    ct.to_csv(rundir / "val_readouts.csv", index=False)
    tracks = [day_tracks(net, dev, cfg, d, rows, Xd)[0] for d, rows, Xd in day_sets if len(Xd)]
    (pd.concat(tracks) if tracks else pd.DataFrame()).to_csv(rundir / "tracks.csv", index=False)
    if len(Xi):
        pmi, _, profi = predict(net, Xi, dev, profile=cfg.profile)
        np.savez_compressed(rundir / "fixed_set_preds.npz", pred=pmi, target=Yi.numpy(), x=Xi.numpy(),
                            prof_true=Pi.numpy(), prof_pred=(profi if profi is not None else np.zeros(0)),
                            path=np.array(img_rows.path.values, dtype=object), title=np.array(img_titles, dtype=object))
    summary = {"run": cfg.run, "stage": cfg.stage, "variant": cfg.variant, "n_train": int(len(tr)), "n_val": int(len(va)),
               "params": n_params(net), "time_total_s": time.time() - t_all,
               "best_epoch": best["epoch"], "best_metric": cfg.best_metric, "best_value": best["value"],
               "best": (hist[best["epoch"]] if best["epoch"] >= 0 else {}), "last": last_m}
    (rundir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    hp = {k: asdict(cfg)[k] for k in ("variant", "depth", "base", "norm", "dropout", "skip", "profile", "lam", "lam_prof",
                                      "lr", "sched", "batch", "seed", "epochs", "render_train", "render_val", "bg_mode")}
    hm = {f"hparam/{k.split('/')[-1]}": float(summary["best"].get(k, np.nan)) for k in
          ("val/IoU_F2", "val/foF2_med", "val/foF2_rmse", "val/hmF2_rmse", "val/prof_rmse", "val/logic_total", "gate/violations")
          if k in summary["best"]}
    try:
        log.w.add_hparams(hp, hm, run_name=".")
    except Exception as e:
        print("  hparams не записаны:", e, flush=True)
    log.close()
    print(f"[{cfg.stage}/{cfg.run}] готово за {time.time() - t_all:.0f} с → {rundir}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(TrainConfig):
        if f.type is bool or f.type == "bool":
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    cfg = TrainConfig(**vars(ap.parse_args()))
    train(cfg)


if __name__ == "__main__":
    main()
