# -*- coding: utf-8 -*-
"""
training.py — тренировочный цикл ВЗ-модели (этап E1: Э3 §2 таблица, §3 метрики, §3.5 наборы
логирования, §3.6 стратификация, §4 протокол; CLAUDE.md §2.3 dry → smoke → full, §2.4 логи).

Раны E1 (VARIANTS): baseline (CE) | +hinge (CE + λ·L_logic hinge) | +lognorm (softplus) |
+coords (координатные каналы, абляция). Данные — потоковый лоадер по манифесту (Э3 §2.2):
train — DataLoader с воркерами; фиксированный val-поднабор декодируется один раз в RAM.

Каждая эпоха: обучение (AMP) → оценка на val-поднаборе: CE; hinge-мера L_logic покомпонентно
(единая для всех ранов) + гистограмма по образцам; IoU по классам, precision/recall следовых
пикселей; |ΔfoF2|, |Δh′F|, |ΔfoE|, |ΔfoEs| против ARTIST (жёсткие ридауты canon) — RMSE,
медиана, доли «точно по РД» / «сомнительно» (РД 52.26.817 §6.5.5: 0.1 МГц|2 % и 0.2 МГц|5 %);
стратификация (станция, день/ночь, C-level ≤22/>22, спокойные/возмущённые); SHACL-гейт на
gate_n сценах (+ ARTIST-референс один раз); панели ионограмм на фиксированном наборе;
суточные треки foF2(t) на фиксированных сутках (RMSE и корреляция хода, Э3 §3.5).
Артефакты runs/<этап>/<ран>/: events (TensorBoard), metrics.csv (строка на эпоху, все скаляры),
weights.pt (ЛУЧШАЯ эпоха по best_metric, по умолчанию val/foF2_med) и weights_last.pt (последняя;
оба — state_dict + cfg + метрики эпохи), config.json (конфиг + provenance: git-коммит, md5
манифеста, версии torch/CUDA/cuDNN/python, GPU, argv, время старта), val_readouts.csv (ридауты
лучшей модели по val-поднабору: foF2/h′F/foE/foEs pred vs ARTIST, L_logic по образцам),
tracks.csv (суточные треки лучшей модели на фиксированных сутках), fixed_set_preds.npz (вход,
цель, предсказание фиксированного набора панелей), summary.json (best и last эпохи).
Набор логирования — runs/<этап>/logging_set.json (создаётся ОДИН раз и не меняется).

Запуск:  python -m pyon.training --stage E1 --run baseline --variant baseline --manifest data/manifest.csv
         python -m pyon.training --dry            (1 шаг, 8 val, 2 сцены; проверка форм и путей)
         python -m pyon.training --stage smoke --run hinge --variant hinge --manifest data/manifest_smoke.csv \\
                --limit 280 --epochs 1 --gate_n 20
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
from pyon import canon, gates, loader, logic, manifest, tblog   # noqa: E402
from pyon import validate as vd                                   # noqa: E402
from pyon.models import UNet, n_params                            # noqa: E402

VARIANTS = {"baseline": dict(logic=None, coords=False), "hinge": dict(logic="hinge", coords=False),
            "lognorm": dict(logic="lognorm", coords=False), "coords": dict(logic=None, coords=True)}
CE_WEIGHTS = [0.05, 1.0, 1.0, 1.0, 1.0]
READOUTS = {"foF2": ("F2", "fmax"), "hF": ("F2", "hmin"), "foE": ("E", "fmax"), "foEs": ("Es", "fmax")}


@dataclass
class TrainConfig:
    run: str = "baseline"
    stage: str = "E1"
    manifest: str = "data/manifest.csv"
    variant: str = "baseline"       # baseline | hinge | lognorm | coords
    lam: float = 0.5                # вес L_logic
    epochs: int = 6
    batch: int = 64
    lr: float = 2e-3
    amp: bool = True
    workers: int = 4
    seed: int = 0
    limit: int = 0                  # smoke: первые N строк манифеста (0 = все)
    val_size: int = 4000            # фиксированный val-поднабор (в RAM)
    gate_n: int = 150               # сцен на SHACL-гейт
    gate_every: int = 5             # гейт каждые N эпох и в последнюю (1.5 с/сцену на ядро)
    gate_procs: int = 4             # процессов для гейта
    log_images: int = 48            # максимум панелей ионограмм из фиксированного набора (по станциям)
    images_every: int = 1           # каждые N эпох
    max_steps: int = 0              # dry: шагов в эпохе (0 = все)
    holdout: str = ""               # leave-one-station-out: станция исключается из train
    block_shuffle: int = 0          # >0: loader.BlockShuffleSampler(block) — если корпус не влезает в page cache
    # архитектура U-Net (дефолт = прототип, 117 605 параметров; исследование — DIARY 2026-09-05)
    depth: int = 3
    base: int = 16
    norm: str = "batch"             # batch | group | none
    dropout: float = 0.0
    skip: bool = True
    best_metric: str = "val/foF2_med"   # критерий лучшей эпохи (меньше — лучше); weights.pt = лучшая
    dry: bool = False
    device: str = "cuda"


# ---------------------------------------------------------------------------- данные
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


def load_split(cfg: TrainConfig):
    df = pd.read_csv(ROOT / cfg.manifest)
    if cfg.limit:   # smoke/dry/подмножество: по половине лимита из каждого сплита, РАВНОМЕРНО по манифесту
        parts = []  # (манифест отсортирован по станции и времени → пропорционально станциям и сезонам)
        for _, g in df.groupby("split"):
            n = min(len(g), max(1, cfg.limit // 2))
            parts.append(g.iloc[np.unique(np.linspace(0, len(g) - 1, n).round().astype(int))])
        df = pd.concat(parts)
    df = df.reset_index(drop=True)
    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    if cfg.holdout:
        tr = tr[tr.station != cfg.holdout]
    # val-поднабор: поровну по станциям, равномерно по времени внутри станции (детерминированно)
    n_st = max(1, va.station.nunique())
    per = max(1, cfg.val_size // n_st)
    parts = []
    for _, g in va.groupby("station"):
        idx = np.unique(np.linspace(0, len(g) - 1, min(per, len(g))).round().astype(int))
        parts.append(g.iloc[idx])
    va_sub = pd.concat(parts).sort_values(["station", "time"]).reset_index(drop=True)
    return df, tr.reset_index(drop=True), va_sub


def decode(df: pd.DataFrame, workers: int, batch: int = 128):
    """Декодировать все пары df в RAM: X uint8 [N,2,H,W], Y int8 [N,H,W]."""
    if not len(df):
        return torch.zeros((0, 2, canon.NH, canon.NF), dtype=torch.uint8), torch.zeros((0, canon.NH, canon.NF), dtype=torch.int8)
    dl = DataLoader(loader.VerticalDataset(df), batch_size=batch, shuffle=False, num_workers=workers)
    xs, ys = zip(*[(x, y) for x, y in dl])
    return torch.cat(xs), torch.cat(ys)


def to_input(x: torch.Tensor, dev) -> torch.Tensor:
    return x.to(dev, non_blocking=True).float().div_(255)


def select_logging_set(va: pd.DataFrame, path: Path, per_cell: int = 2) -> dict:
    """Э3 §3.5: (а) на станцию по per_cell ионограмм × {день, ночь} × {спокойный, возмущённый};
    (б) на станцию по двое фиксированных суток (спокойные + возмущённые) с ≥ 24 ионограммами.
    Создаётся ОДИН раз (файл path) и далее не меняется между ранами и этапами."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    va = va.copy()
    va["dn"] = [manifest.daynight(s, t) for s, t in zip(va.station, va.time)]
    va["date"] = pd.to_datetime(va.time).dt.date.astype(str)
    images, days = [], []
    for st, g in va.groupby("station"):
        for dn in ("day", "night"):
            for dist in (0, 1):
                cell = g[(g.dn == dn) & (g.disturbed == dist)].sort_values("time").head(per_cell)
                images += [dict(path=r.path, station=st, time=r.time, dn=dn, disturbed=int(dist)) for r in cell.itertuples()]
        by_day = g.groupby("date").agg(n=("path", "size"), dist=("disturbed", "sum")).query("n >= 24")
        if len(by_day):
            quiet = by_day[by_day.dist == 0]
            days.append(dict(station=st, date=(quiet if len(quiet) else by_day).index[0], kind="quiet"))
            dist_days = by_day[by_day.dist >= 3]
            if len(dist_days):
                days.append(dict(station=st, date=dist_days.dist.idxmax(), kind="disturbed"))
    out = dict(images=images, days=days, created=time.strftime("%Y-%m-%d %H:%M"), manifest_rows=int(len(va)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------- метрики
def readouts_table(pm: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Жёсткие ридауты по маскам против ARTIST (колонки манифеста)."""
    rows = []
    for k in range(len(pm)):
        r = {}
        for name, (cls, kind) in READOUTS.items():
            ci = canon.CLASSES.index(cls)
            r[f"{name}_pred"] = canon.fmax_readout(pm[k], ci) if kind == "fmax" else canon.hmin_readout(pm[k], ci)
            r[f"{name}_artist"] = float(meta[name].iloc[k]) if name in meta else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def err_stats(pred, true, fo_true=None) -> dict:
    """RMSE, медиана |Δ|, n; для foF2 — доли «точно по РД» (0.1 МГц | 2 %) и «сомнительно» (0.2 | 5 %)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(pred) & np.isfinite(true)
    out = dict(n=int(ok.sum()))
    if not ok.any():
        return out
    d = np.abs(pred[ok] - true[ok])
    out.update(rmse=float(np.sqrt(np.mean(d ** 2))), med=float(np.median(d)),
               n_pred=int(np.isfinite(pred).sum()), n_true=int(np.isfinite(true).sum()))
    if fo_true is not None:
        t = np.asarray(fo_true, float)[ok]
        out["rd_exact"] = float(np.mean(d <= np.maximum(0.1, 0.02 * t)))
        out["rd_doubt"] = float(np.mean(d <= np.maximum(0.2, 0.05 * t)))
    return out


def strata(meta: pd.DataFrame) -> dict:
    """Э3 §3.6: станция; день/ночь; C-level ≤ 22 / > 22; спокойные/возмущённые."""
    dn = np.array([manifest.daynight(s, t) for s, t in zip(meta.station, meta.time)])
    out = {f"station_{st}": (meta.station.values == st) for st in sorted(meta.station.unique())}
    out.update({"day": dn == "day", "night": dn == "night",
                "clevel_le22": meta.c_level.values <= 22, "clevel_gt22": meta.c_level.values > 22,
                "quiet": meta.disturbed.values == 0, "disturbed": meta.disturbed.values == 1})
    return {k: v for k, v in out.items() if v.sum() > 0}


@torch.no_grad()
def predict(net, X: torch.Tensor, dev, batch: int = 128):
    """Логиты не храним: возвращает argmax-маски [N,H,W] int8 и hinge-L_logic по образцам/компонентам."""
    net.eval()
    pms, comps = [], {}
    if not len(X):
        return np.zeros((0, canon.NH, canon.NF), np.int8), {}
    for k in range(0, len(X), batch):
        lg = net(to_input(X[k:k + batch], dev)).float()
        pms.append(lg.argmax(1).to(torch.int8).cpu())
        _, c = logic.vertical_logic(lg, "hinge")
        for name, v in c.items():
            comps.setdefault(name, []).append(v.detach().cpu())
    pm = torch.cat(pms).numpy()
    comps = {k: torch.cat(v).numpy() for k, v in comps.items()}
    return pm, comps


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


def evaluate(net, X, Y, meta, dev, cfg: TrainConfig, vocab, epoch: int, log: tblog.TBLog, artist_gate: dict):
    """Полная оценка на фиксированном val-поднаборе → dict скаляров (ключи как в TensorBoard)."""
    m = {}
    m["val/CE"] = val_ce(net, X, Y, dev)
    pm, comps = predict(net, X, dev)
    Yn = Y.numpy()
    for k, v in comps.items():
        m[f"val/logic_{k}"] = float(v.mean())
    tot = sum(comps.values())
    m["val/logic_total"] = float(tot.mean())
    log.hist("val/logic_total_hist", tot, epoch)
    m.update({f"val/{k}": v for k, v in segmentation_metrics(pm, Yn).items()})
    rt = readouts_table(pm, meta)
    for name in READOUTS:
        st = err_stats(rt[f"{name}_pred"], rt[f"{name}_artist"], rt["foF2_artist"] if name == "foF2" else None)
        for k, v in st.items():
            m[f"val/{name}_{k}"] = v
    for sname, mask in strata(meta).items():
        if mask.sum() < 5:
            continue
        st = err_stats(rt.foF2_pred[mask], rt.foF2_artist[mask])
        m[f"strat/{sname}/foF2_rmse"] = st.get("rmse", np.nan)
        m[f"strat/{sname}/foF2_med"] = st.get("med", np.nan)
        m[f"strat/{sname}/IoU_F2"] = segmentation_metrics(pm[mask], Yn[mask])["IoU_F2"]
        m[f"strat/{sname}/n"] = int(mask.sum())
    # SHACL-гейт (Э3 §3.3): первые gate_n сцен val, каждые gate_every эпох и в последнюю;
    # ARTIST-референс — один раз
    n_g = min(cfg.gate_n, len(pm))
    if n_g and (epoch % cfg.gate_every == 0 or epoch == cfg.epochs - 1):
        t_g = time.time()
        rate, _ = gates.gate_rate(pm[:n_g], gates.vertical_scene, vocab, prefix=f"e{epoch}_", procs=cfg.gate_procs)
        m["gate/violations"] = rate
        if "artist" not in artist_gate:
            artist_gate["artist"], _ = gates.gate_rate(Yn[:n_g], gates.vertical_scene, vocab, prefix="art_", procs=cfg.gate_procs)
        m["gate/artist_violations"] = artist_gate["artist"]
        m["gate/n"] = n_g
        m["time/gate_s"] = time.time() - t_g
    rt.insert(0, "logic_total", tot)
    return m, pm, rt


# ---------------------------------------------------------------------------- цикл
def train(cfg: TrainConfig) -> dict:
    if cfg.dry:
        cfg.limit = cfg.limit or 48; cfg.epochs = 1; cfg.max_steps = 1; cfg.val_size = 8
        cfg.gate_n = 2; cfg.log_images = 2; cfg.workers = min(cfg.workers, 2); cfg.gate_procs = 2
        cfg.stage, cfg.run = "dry", cfg.run
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
    dl = DataLoader(loader.VerticalDataset(tr), batch_size=cfg.batch, shuffle=sampler is None, sampler=sampler,
                    num_workers=cfg.workers, persistent_workers=cfg.workers > 0,
                    prefetch_factor=4 if cfg.workers > 0 else None,
                    drop_last=len(tr) >= cfg.batch, pin_memory=dev.type == "cuda")
    t0 = time.time()
    Xv, Yv = decode(va, cfg.workers)
    print(f"  val-поднабор декодирован: X {tuple(Xv.shape)} {Xv.dtype}, Y {tuple(Yv.shape)} {Yv.dtype}, "
          f"{time.time() - t0:.1f} с", flush=True)
    # фиксированный набор логирования (Э3 §3.5) — общий для этапа
    lset = select_logging_set(df[df.split == "val"], ROOT / "runs" / cfg.stage / "logging_set.json")
    img_rows = df[df.path.isin([im["path"] for im in lset["images"]])].sort_values(["station", "time"]).head(cfg.log_images)
    Xi, Yi = decode(img_rows, min(cfg.workers, 2))
    img_titles = [f"{r.station} {str(r.time)[5:16]} {manifest.daynight(r.station, r.time)}"
                  + (" ВОЗМ" if r.disturbed else "") for r in img_rows.itertuples()]
    img_groups = {st: np.flatnonzero(img_rows.station.values == st) for st in img_rows.station.unique()}
    day_sets = []
    for d in lset["days"]:
        rows = df[(df.station == d["station"]) & (pd.to_datetime(df.time).dt.date.astype(str) == d["date"])].sort_values("time")
        Xd, _ = decode(rows, min(cfg.workers, 2))
        day_sets.append((d, rows.reset_index(drop=True), Xd))
    print(f"  набор логирования: {len(img_rows)} панелей, {len(day_sets)} суток-треков", flush=True)

    net = UNet(2, len(canon.CLASSES), base=cfg.base, depth=cfg.depth, norm=cfg.norm, dropout=cfg.dropout,
               skip=cfg.skip, coords=v["coords"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), cfg.lr)
    scaler = torch.amp.GradScaler(enabled=(cfg.amp and dev.type == "cuda"))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(CE_WEIGHTS, device=dev))
    vocab = vd.load_vocabulary()
    artist_gate: dict = {}
    print(f"  U-Net {n_params(net)} параметров (depth {cfg.depth}, base {cfg.base}, norm {cfg.norm}, dropout {cfg.dropout}, "
          f"skip {cfg.skip}), вариант {cfg.variant} (logic={v['logic']}, coords={v['coords']}), AMP={scaler.is_enabled()}", flush=True)
    best = dict(value=float("inf"), epoch=-1)

    hist = []
    for ep in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        net.train(); t_ep = time.time(); n_seen = 0
        sums = {"CE": 0.0, "logic": 0.0}; csum = {}
        for step, (x, y) in enumerate(dl):
            if cfg.max_steps and step >= cfg.max_steps:
                break
            x, y = to_input(x, dev), y.to(dev, non_blocking=True).long()
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                lg = net(x)
                loss_ce = ce(lg, y)
            loss = loss_ce
            if v["logic"]:
                l_logic, comps = logic.vertical_logic(lg.float(), v["logic"])
                loss = loss + cfg.lam * l_logic
                sums["logic"] += l_logic.item() * len(x)
                for k, c in comps.items():
                    csum[k] = csum.get(k, 0.0) + c.mean().item() * len(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sums["CE"] += loss_ce.item() * len(x); n_seen += len(x)
            if cfg.dry:
                print(f"  dry: batch x {tuple(x.shape)} y {tuple(y.shape)} logits {tuple(lg.shape)} "
                      f"CE {loss_ce.item():.3f} loss {loss.item():.3f}", flush=True)
        t_train = time.time() - t_ep
        m = {"train/CE": sums["CE"] / max(n_seen, 1), "time/train_s": t_train,
             "time/train_samples_per_s": n_seen / max(t_train, 1e-9), "epoch": ep}
        if v["logic"]:
            m["train/logic"] = sums["logic"] / max(n_seen, 1)
            m.update({f"train/logic_{k}": s_ / max(n_seen, 1) for k, s_ in csum.items()})
        t_ev = time.time()
        mv, pm, rt = evaluate(net, Xv, Yv, va, dev, cfg, vocab, ep, log, artist_gate)
        m.update(mv)
        # панели и треки
        if len(Xi) and (ep % cfg.images_every == 0 or ep == cfg.epochs - 1):
            pmi, _ = predict(net, Xi, dev)
            for st, ix in img_groups.items():          # одна фигура на станцию (Э3 §3.5-а)
                log.ionograms(f"images/{st}", Xi.numpy()[ix], Yi.numpy()[ix], pmi[ix], ep,
                              titles=[img_titles[i] for i in ix])
        for d, rows, Xd in day_sets:
            if not len(Xd):
                continue
            pmd, _ = predict(net, Xd, dev)
            fo_m = np.array([canon.fmax_readout(pmd[k], 1) for k in range(len(pmd))])
            fo_a = rows.foF2.values.astype(float)
            hours = pd.to_datetime(rows.time).dt.hour.values + pd.to_datetime(rows.time).dt.minute.values / 60
            key = f"{d['station']}_{d['date']}_{d['kind']}"
            log.track(f"track/{key}", hours, {"ARTIST": fo_a, cfg.run: fo_m}, ep, title=key)
            st = err_stats(fo_m, fo_a)
            ok = np.isfinite(fo_m) & np.isfinite(fo_a)
            m[f"track/{key}/rmse"] = st.get("rmse", np.nan)
            m[f"track/{key}/corr"] = float(np.corrcoef(fo_m[ok], fo_a[ok])[0, 1]) if ok.sum() > 2 else np.nan
        m["time/eval_s"] = time.time() - t_ev
        log.scalars({k: v_ for k, v_ in m.items() if k != "epoch"}, ep)
        log.row(m, ep); hist.append(m)
        ckpt = {"state_dict": net.state_dict(), "cfg": asdict(cfg), "epoch": ep, "metrics": m}
        torch.save(ckpt, rundir / "weights_last.pt")
        crit = m.get(cfg.best_metric, np.nan)
        if np.isfinite(crit) and crit < best["value"]:
            best = dict(value=float(crit), epoch=ep)
            torch.save(ckpt, rundir / "weights.pt")          # weights.pt = лучшая эпоха по best_metric
        print(f"  ep{ep}: train CE {m['train/CE']:.3f}" + (f" logic {m['train/logic']:.4f}" if v["logic"] else "")
              + f" | val CE {m['val/CE']:.3f} L_hinge {m['val/logic_total']:.4f} IoU_F2 {m.get('val/IoU_F2', np.nan):.3f} "
              f"foF2 RMSE {m.get('val/foF2_rmse', np.nan):.2f} med {m.get('val/foF2_med', np.nan):.2f} "
              f"gate {m.get('gate/violations', np.nan):.0%} (ARTIST {m.get('gate/artist_violations', np.nan):.0%}) "
              f"| {t_train:.0f}+{m['time/eval_s']:.0f} с, {m['time/train_samples_per_s']:.0f} обр/с", flush=True)

    # финальные артефакты — от ЛУЧШИХ весов (weights.pt), чтобы csv/npz соответствовали сохранённой модели
    last_m = hist[-1]
    if best["epoch"] >= 0 and best["epoch"] != cfg.epochs - 1:
        net.load_state_dict(torch.load(rundir / "weights.pt", map_location=dev)["state_dict"])
        pm, comps = predict(net, Xv, dev)
        rt = readouts_table(pm, va); rt.insert(0, "logic_total", sum(comps.values()))
    rt.insert(0, "time", va.time.values); rt.insert(0, "station", va.station.values); rt.insert(0, "path", va.path.values)
    rt.to_csv(rundir / "val_readouts.csv", index=False)
    tracks = []
    for d, rows, Xd in day_sets:
        if not len(Xd):
            continue
        pmd, _ = predict(net, Xd, dev)
        for k in range(len(pmd)):
            tracks.append(dict(station=d["station"], date=d["date"], kind=d["kind"], time=rows.time.iloc[k],
                               foF2_artist=float(rows.foF2.iloc[k]), foF2_pred=canon.fmax_readout(pmd[k], 1),
                               hF_artist=float(rows.hF.iloc[k]), hF_pred=canon.hmin_readout(pmd[k], 1)))
    pd.DataFrame(tracks).to_csv(rundir / "tracks.csv", index=False)
    if len(Xi):
        pmi, _ = predict(net, Xi, dev)
        np.savez_compressed(rundir / "fixed_set_preds.npz", pred=pmi, target=Yi.numpy(), x=Xi.numpy(),
                            path=np.array(img_rows.path.values, dtype=object), title=np.array(img_titles, dtype=object))
    summary = {"run": cfg.run, "stage": cfg.stage, "variant": cfg.variant, "n_train": int(len(tr)), "n_val": int(len(va)),
               "params": n_params(net), "time_total_s": time.time() - t_all,
               "best_epoch": best["epoch"], "best_metric": cfg.best_metric, "best_value": best["value"],
               "best": (hist[best["epoch"]] if best["epoch"] >= 0 else {}), "last": last_m}
    (rundir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
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
