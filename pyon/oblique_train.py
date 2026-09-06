# -*- coding: utf-8 -*-
"""
oblique_train.py — НЗ-модель на синтетике (этап E5; Э2 §3, §5; Э3 §2 E3/E5, §3.1–3.3, §3.5–3.6).

Данные (Э3 §2.2 E3): `loader.ObliqueDataset` синтезирует маску НЗ на лету из SAO — следы ARTIST
пересчитаны секансом на сферической Земле (Брейт–Тьюв, Мартин; Э2 §2.4), O-компонента (X — E3b),
дальность D случайна из D_SET на каждый показ, классы obs.OB_CLASSES (BG F2 F1 E Es MH — кратник
2F2), точные МПЧ-метки мод. Сырьё порождает нейрорендерер (E4, `pyon.renderer`) на GPU в батче:
новая реализация спекла на каждом показе (бесконечная аугментация).

Модель: U-Net(2 → 6) той же архитектуры, что E1. Раны: baseline (CE) | +lognorm / +hinge
(CE + λ·L_logic: S1 МПЧ(2F2) ≤ МПЧ(1F2), S2 P′(2F2) ≥ P′(1F2)+150 км, P2 связность — с детачем).

Метрики (Э3 §3): IoU по классам (в т.ч. MH); |ΔМПЧ| RMSE/медиана для 1F2 и 2F2 против аналитических
меток; P′ носа; инвариант Пономарчука МПЧ(2F2)/МПЧ(1F2) (медиана/IQR предсказаний vs меток +
гистограмма); SHACL-гейт НЗ-сцен (`gates.oblique_scene`) + референс меток (обязан быть 0 %);
стратификация по D; кросс-тест «ВЗ-модель zero-shot на НЗ» (`--vz_weights`); суточные треки
МПЧ(t) на фиксированных сутках E1 при фиксированной D.

Запуск:  python -m pyon.oblique_train --stage E5 --run lognorm --variant lognorm \\
             --renderer runs/E4/base/weights.pt --manifest data/manifest.csv --epochs 5
         python -m pyon.oblique_train --dry --renderer runs/dry/dry_render/weights.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import canon, gates, loader, logic, renderer as rnd, tblog, training   # noqa: E402
from pyon import digi_formats as dfm                                                # noqa: E402
from pyon import oblique_synth as obs                                               # noqa: E402
from pyon import validate as vd                                                     # noqa: E402
from pyon.models import UNet, n_params                                              # noqa: E402

VARIANTS = {"baseline": None, "lognorm": "lognorm", "hinge": "hinge"}
CE_WEIGHTS = [0.05, 1.0, 1.0, 1.0, 1.0, 1.0]
EXTENT = (obs.FOB_MIN, obs.FOB_MAX, obs.P_MIN, obs.P_MAX)
iF2, iMH, iEs = obs.OB_CLASSES.index("F2"), obs.OB_CLASSES.index("MH"), obs.OB_CLASSES.index("Es")


@dataclass
class ObliqueConfig:
    run: str = "lognorm"
    stage: str = "E5"
    manifest: str = "data/manifest.csv"
    renderer: str = "runs/E4/base/weights.pt"
    variant: str = "lognorm"        # baseline | lognorm | hinge
    lam: float = 0.5
    component: str = "O"
    epochs: int = 5
    steps_per_epoch: int = 0        # 0 = весь train-манифест
    batch: int = 64
    lr: float = 2e-3
    sched: str = "cosine"
    depth: int = 4
    base: int = 32
    amp: bool = True
    workers: int = 8
    seed: int = 0
    limit: int = 0
    val_size: int = 2400
    gate_n: int = 150
    gate_every: int = 2
    gate_procs: int = 4
    log_images: int = 16
    images_every: int = 1
    track_d: float = 800.0          # дальность для суточных треков МПЧ
    vz_weights: str = ""            # кросс-тест: ВЗ-модель zero-shot
    max_steps: int = 0
    dry: bool = False
    device: str = "cuda"
    render_corr: str = "9,3"        # длина корреляции спекла рендерера "k_h,k_f" (см. training.TrainConfig)
    dataset: str = ""               # материализованный НЗ-датасет (pyon.oblique_dataset, шарды npz); "" — синтез на лету
    input_mode: str = "ox"          # "ox" — вход = суммарная мощность O + X в одном канале (как видит чирп-зонд,
                                    # поляризации не разделены); "o" — только O (режим раунда 1)
    tromso_route: str = "sgo-tgo"   # контроль на РЕАЛЬНЫХ НЗ каждый eval (без меток: доля найденных F2, МПЧ, гейт)
    tromso_n: int = 24              # сколько последних снимков брать (0 — не проверять)
    tromso_active: float = 0.05     # целевая доля активных пикселей при растеризации снимков (pyon.tromso)
    cover: float = 0.7              # доля образцов со случайным ОКНОМ ПОКРЫТИЯ (вне окна нули — как на
                                    # реальном снимке, где поле уже нашей решётки)
    density: str = "0.03,0.18"      # случайная целевая доля активных пикселей: синтетика проходит ТУ ЖЕ
                                    # нормировку, что реальные снимки (порог по квантилю); "" — выключить
    bg_shift: bool = True           # фон вне следов — из отдельного рендера пустой маски со случайным
                                    # сдвигом по частоте (рендерер обучен на ВЗ-сетке, полосы иначе стоят
                                    # в тех же столбцах = на бессмысленных для НЗ частотах)


def decode_oblique(df: pd.DataFrame, component: str, workers: int, seed: int = 0, fixed_d=None, batch: int = 128):
    """ObliqueDataset → Y int8 [N,H,W], D [N], muf_F2 [N], muf_MH [N] (в RAM)."""
    if not len(df):
        z = torch.zeros(0)
        return torch.zeros((0, obs.NP, obs.NF), dtype=torch.int8), z, z, z
    ds = loader.ObliqueDataset(df, component=component, fixed_d=fixed_d, seed=seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    ys, ds_, m1, m2 = zip(*[b for b in dl])
    return torch.cat(ys), torch.cat(ds_), torch.cat(m1), torch.cat(m2)


def _dens(cfg):
    return tuple(float(v) for v in cfg.density.split(",")) if cfg.density else None


@torch.no_grad()
def render_input(ren, mh2f2, yo: torch.Tensor, yx: torch.Tensor | None, mode: str = "ox",
                 bg_shift: bool = True, cover: float = 0.7, density=(0.03, 0.18)) -> torch.Tensor:
    """Вход НЗ-модели [B,2,H,W]: канал 0 — суммарная мощность (рендер O-следов + X-следы из рендера
    X-маски, ограниченные расширенной X-маской), канал 1 — нули. Чирп-зонд не разделяет поляризаций:
    на реальном снимке O- и X-следы лежат в одной картинке (`pyon.tromso` даёт тот же формат).

    bg_shift: фон вне расширенной маски следов берётся у ДОНОРА (другого образца того же батча),
    циклически сдвинутого по частоте на случайную величину; след донора закрывается собственным
    фоном. Зачем: рендерер обучен на ВЗ-сетке (1–15 МГц), НЗ-сетка другая (2–24 МГц) — без сдвига
    помеховые полосы садятся в те же СТОЛБЦЫ, то есть на фиксированные и бессмысленные для НЗ
    частоты (галерея датасета 2026-09-06). Фон донора берётся из ТОГО ЖЕ рендерера, поэтому
    текстурного шва нет (в отличие от трансплантации РЕАЛЬНОГО фона, опровергнутой в E4).
    Пустая маска на роль фона не годится: рендерер выучил, что «нет разметки ARTIST» = возмущённая
    сильно зашумлённая ионограмма, и рисует именно её."""
    x = ren.sample(mh2f2[yo])[:, :1]
    if mode == "ox" and yx is not None:
        xx = ren.sample(mh2f2[yx])[:, :1]
        reg_x = Fn.max_pool2d((yx > 0).float().unsqueeze(1), 5, stride=1, padding=2)
        x = (x + xx * reg_x).clamp(max=1.0)
    m = (yo > 0) | (yx > 0) if yx is not None else (yo > 0)
    reg = Fn.max_pool2d(m.float().unsqueeze(1), 7, stride=1, padding=3) > 0
    if cover > 0:
        # ОКНО ПОКРЫТИЯ: реальный НЗ-снимок покрывает лишь часть нашей решётки (SGO→TGO: 2–16 МГц ×
        # 300–1500 км; Juliusruh→TGO: 2–14 МГц × 1500–3200 км), вне окна данных нет — нули. Синтетика
        # без окна заполняет всё поле: доля активных 15 % против 1 % у реального (сравнение 2026-09-06).
        # Окно применяется к образцу, только если след целиком в него попадает (иначе метка требовала бы
        # угадать невидимое).
        B, _, H, W = x.shape
        take = torch.rand(B, device=x.device) < cover
        f_hi = torch.randint(int(0.45 * W), W + 1, (B,), device=x.device)      # верхняя частота окна
        p_hi = torch.randint(int(0.35 * H), H + 1, (H and B,), device=x.device)  # верхний групповой путь
        cols = torch.arange(W, device=x.device).view(1, 1, 1, W)
        rows = torch.arange(H, device=x.device).view(1, 1, H, 1)
        win = (cols < f_hi.view(B, 1, 1, 1)) & (rows < p_hi.view(B, 1, 1, 1))
        fits = (m.unsqueeze(1) & ~win).flatten(1).sum(1) == 0                   # след целиком внутри окна
        win = win | (~(take & fits)).view(B, 1, 1, 1)
        x = x * win.float()
    if bg_shift and len(x) > 1:
        k = int(torch.randint(0, x.shape[-1], (1,)).item())
        donor = torch.roll(torch.roll(x, 1, 0), shifts=k, dims=-1)          # сосед по батчу, сдвиг по частоте
        donor_reg = torch.roll(torch.roll(reg, 1, 0), shifts=k, dims=-1)
        donor = torch.where(donor_reg, x, donor)                            # след донора закрываем своим фоном
        x = torch.where(reg, x, donor)
    if density is not None:
        # ТА ЖЕ нормировка, что у реальных снимков (`tromso.rasterize`): порог по квантилю (1 − a) с
        # СЛУЧАЙНОЙ целевой долей активных a, растяжение по 99.9-му перцентилю. Так train и test
        # проходят один оператор, а модель не привязывается к плотности помех домена дигизонда
        # (у чирп-зонда Тромсё поле заметно чище: сравнение 2026-09-06).
        B = x.shape[0]
        n_win = win.flatten(1).sum(1).float() if cover > 0 else torch.full((B,), float(x[0].numel()), device=x.device)
        a = torch.empty(B, device=x.device).uniform_(density[0], density[1])
        flat = x.flatten(1)
        sv, _ = torch.sort(flat, dim=1, descending=True)
        kth = (a * n_win).long().clamp(1, flat.shape[1] - 1)
        thr = sv.gather(1, kth.unsqueeze(1))
        top = sv[:, :max(1, flat.shape[1] // 1000)].mean(1, keepdim=True)
        x = ((x - thr.view(B, 1, 1, 1)) / (top - thr).clamp(min=1e-3).view(B, 1, 1, 1)).clamp(0, 1)
    return torch.cat([x, torch.zeros_like(x)], 1)


@torch.no_grad()
def tromso_probe(net, dev, cfg, vocab=None, do_gate: bool = False) -> dict:
    """Контроль на РЕАЛЬНЫХ НЗ Тромсё без меток (Э3 §4 п.4; критерий ранней остановки для E5, где
    реальных меток нет — вывод E4 о деградации переноса с числом шагов): доля снимков с найденным F2,
    медиана МПЧ 1F2 и её IQR, инвариант 2F2/1F2, доля SHACL-нарушений на реальных сценах."""
    from pyon import tromso as tg
    d = ROOT / "data" / "tromso" / cfg.tromso_route
    files = sorted(d.glob("*.png"))[-cfg.tromso_n:] if cfg.tromso_n else []
    if not files:
        return {}
    xs, cov = [], []
    for fp in files:
        try:
            snr, f, r = tg.png_to_snr(fp, cfg.tromso_route)
            x, cv = tg.rasterize(snr, f, r, "zero", cfg.tromso_active, None)
            xs.append(x); cov.append(cv)
        except Exception:
            continue
    if not xs:
        return {}
    X = torch.from_numpy(np.stack(xs)).float().div(255)
    pm, _ = predict(net, X, dev)
    pm = np.where(np.stack(cov), pm, 0).astype(pm.dtype)
    f1, f2, pn = muf_readouts(pm)
    m = {"real/n": len(xs), "real/has_F2_frac": float(np.isfinite(f1).mean()),
         "real/muf1F2_med": float(np.nanmedian(f1)) if np.isfinite(f1).any() else np.nan,
         "real/muf1F2_iqr": float(np.nanpercentile(f1, 75) - np.nanpercentile(f1, 25)) if np.isfinite(f1).sum() > 3 else np.nan,
         "real/has_MH_frac": float(np.isfinite(f2).mean())}
    ok = np.isfinite(f1) & np.isfinite(f2)
    if ok.any():
        m["real/inv_ratio_med"] = float(np.median(f2[ok] / f1[ok]))
    if do_gate and vocab is not None:
        rate, warn, _ = gates.gate_rate(pm, gates.oblique_scene, vocab, prefix="real_", procs=cfg.gate_procs, with_warnings=True)
        m["real/gate_violations"], m["real/gate_warnings"] = rate, warn
    return m


def muf_readouts(pm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """МПЧ 1F2 (класс F2), МПЧ 2F2 (MH) и P′ носа F2 по маскам."""
    f1 = np.array([canon.fmax_readout(p, iF2, obs.fob_axis) for p in pm])
    f2 = np.array([canon.fmax_readout(p, iMH, obs.fob_axis) for p in pm])
    pn = np.array([canon.hmin_readout(p, iF2, obs.p_axis) for p in pm])
    return f1, f2, pn


@torch.no_grad()
def predict(net, X: torch.Tensor, dev, batch: int = 128):
    net.eval(); pms, comps = [], {}
    for k in range(0, len(X), batch):
        lg = net(X[k:k + batch].to(dev)).float()
        pms.append(lg.argmax(1).to(torch.int8).cpu())
        _, c = logic.oblique_logic(lg, "hinge")
        for name, v in c.items():
            comps.setdefault(name, []).append(v.detach().cpu())
    return torch.cat(pms).numpy(), {k: torch.cat(v).numpy() for k, v in comps.items()}


def evaluate(net, Xv, Yv, Dv, M1, M2, dev, cfg, vocab, epoch, log, ref_gate: dict, vz_net=None):
    m = {}
    pm, comps = predict(net, Xv, dev)
    Yn = Yv.numpy()
    for k, v in comps.items():
        m[f"val/logic_{k}"] = float(v.mean())
    tot = sum(comps.values()); m["val/logic_total"] = float(tot.mean()); log.hist("val/logic_total_hist", tot, epoch)
    for ci, cls in enumerate(obs.OB_CLASSES[1:], 1):
        inter = ((pm == ci) & (Yn == ci)).sum(); union = ((pm == ci) | (Yn == ci)).sum()
        m[f"val/IoU_{cls}"] = float(inter / union) if union else np.nan
    f1, f2, pn = muf_readouts(pm)
    lab1, lab2 = M1.numpy(), M2.numpy()
    for name, pred, lab in (("MUF1F2", f1, lab1), ("MUF2F2", f2, lab2)):
        st = training.err_stats(pred, lab)
        for k, v in st.items():
            m[f"val/{name}_{k}"] = v
    Dn = Dv.numpy()
    for d in sorted(set(Dn.tolist())):
        mk = Dn == d
        st = training.err_stats(f1[mk], lab1[mk]); m[f"strat/D{int(d)}/MUF1F2_rmse"] = st.get("rmse", np.nan)
        inter = ((pm[mk] == iF2) & (Yn[mk] == iF2)).sum(); union = ((pm[mk] == iF2) | (Yn[mk] == iF2)).sum()
        m[f"strat/D{int(d)}/IoU_F2"] = float(inter / union) if union else np.nan
    # инвариант Пономарчука: отношение МПЧ кратностей
    okl = np.isfinite(lab1) & np.isfinite(lab2); okp = np.isfinite(f1) & np.isfinite(f2)
    rl, rp = lab2[okl] / lab1[okl], f2[okp] / f1[okp]
    if len(rl):
        m["val/inv_ratio_label_med"] = float(np.median(rl)); m["val/inv_ratio_label_iqr"] = float(np.percentile(rl, 75) - np.percentile(rl, 25))
    if len(rp):
        m["val/inv_ratio_pred_med"] = float(np.median(rp)); m["val/inv_ratio_pred_iqr"] = float(np.percentile(rp, 75) - np.percentile(rp, 25))
        log.hist("val/inv_ratio_pred", rp, epoch); log.hist("val/inv_ratio_label", rl, epoch)
    n_g = min(cfg.gate_n, len(pm))
    if n_g and (epoch % cfg.gate_every == 0 or epoch == cfg.epochs - 1):
        t_g = time.time()
        rate, warn, _ = gates.gate_rate(pm[:n_g], gates.oblique_scene, vocab, prefix=f"e{epoch}_", procs=cfg.gate_procs, with_warnings=True)
        m["gate/violations"], m["gate/warnings"] = rate, warn
        if "labels" not in ref_gate:
            ref_gate["labels"], _, _ = gates.gate_rate(Yn[:n_g], gates.oblique_scene, vocab, prefix="lab_", procs=cfg.gate_procs, with_warnings=True)
        m["gate/labels_violations"] = ref_gate["labels"]; m["gate/n"] = n_g; m["time/gate_s"] = time.time() - t_g
    if vz_net is not None and "vz" not in ref_gate:      # кросс-тест ВЗ-модели zero-shot (один раз)
        vz_net.eval(); pmv = []
        with torch.no_grad():
            for k in range(0, len(Xv), 128):
                pmv.append(vz_net(Xv[k:k + 128].to(dev)).float().argmax(1).cpu())
        pmv = torch.cat(pmv).numpy()
        fv = np.array([canon.fmax_readout(p, 1, obs.fob_axis) for p in pmv])
        inter = ((pmv == 1) & (Yn == iF2)).sum(); union = ((pmv == 1) | (Yn == iF2)).sum()
        ref_gate["vz"] = dict(IoU_F2=float(inter / max(union, 1)), MUF1F2_rmse=training.err_stats(fv, lab1).get("rmse", np.nan))
    if "vz" in ref_gate:
        m["crosstest/vz_zero_shot_IoU_F2"] = ref_gate["vz"]["IoU_F2"]; m["crosstest/vz_zero_shot_MUF1F2_rmse"] = ref_gate["vz"]["MUF1F2_rmse"]
    return m, pm, pd.DataFrame(dict(D=Dn, MUF1F2_label=lab1, MUF1F2_pred=f1, MUF2F2_label=lab2, MUF2F2_pred=f2, Pnose_pred=pn, logic_total=tot))


def train_oblique(cfg: ObliqueConfig) -> dict:
    if cfg.dry:
        cfg.limit = cfg.limit or 48; cfg.epochs = 1; cfg.max_steps = 1; cfg.val_size = 8; cfg.gate_n = 2
        cfg.log_images = 2; cfg.workers = min(cfg.workers, 2); cfg.gate_procs = 2; cfg.stage = "dry"
    training.set_seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    rundir = ROOT / "runs" / cfg.stage / cfg.run
    prov = dict(device=str(dev), torch=torch.__version__, git_commit=training._git_commit(),
                manifest_md5=training._md5(ROOT / cfg.manifest), argv=" ".join(sys.argv),
                started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    log = tblog.TBLog(rundir, asdict(cfg) | prov)
    log.readme(f"""## {cfg.stage}/{cfg.run} — НЗ-модель на синтетике, вариант **{cfg.variant}**
Вход: синтетическая НЗ-ионограмма 128×128 (2–24 МГц × 300–3200 км групповой путь) = маска ARTIST, пересчитанная сферическим секансом
(D ∈ {{300, 800, 1500}} км), «озвученная» рендерером `{cfg.renderer}`; режим входа **{cfg.input_mode}** ("ox" — суммарная мощность
O + X в одном канале, как видит чирп-зонд; X-следы из fo²=fx(fx−fB) с поправкой высот E3b). Цель: классы O-следов F2/F1/E/Es/MH (кратник)
— X-след для модели помеха, которую нужно не спутать с F2. Данные: {"шарды " + cfg.dataset if cfg.dataset else "синтез на лету из SAO"}.
**SCALARS**: `1_train/*` — лоссы; `2_val/*` — IoU по классам, МПЧ 1F2/2F2 RMSE и медиана (МГц, против аналитических меток),
inv_ratio_pred_med vs inv_ratio_label_med — инвариант Пономарчука МПЧ(2F2)/МПЧ(1F2) (должны совпадать), logic_total — нарушения физики;
`3_gate/*` — SHACL-нарушения на инференсе (labels_violations — референс меток, ожидание 0).
**TEXT**: `tables/strat` — по дальностям D; `tables/track` — суточные треки МПЧ(t) при D = {cfg.track_d:.0f} км; `tables/crosstest` — ВЗ-модель zero-shot на НЗ (должна быть много хуже).
**IMAGES**: `images/fixed_set` — вход | метка | предсказание (фиксированный набор, по D); `track/*` — суточный ход МПЧ 1F2/2F2 (метки точки, модель линия).
**HISTOGRAMS**: инвариант Пономарчука (предсказание vs метки).
**`real/*` — контроль на РЕАЛЬНЫХ НЗ Тромсё ({cfg.tromso_route}, последние {cfg.tromso_n} снимков, меток нет)**: has_F2_frac — доля снимков
с найденным следом F2, muf1F2_med/iqr — медиана и разброс МПЧ (МГц), inv_ratio_med — инвариант 2F2/1F2, gate_violations — доля SHACL-нарушений
на реальных сценах. Это единственный критерий переноса без меток (вывод E4: перенос рендер→реальное деградирует с числом шагов) — ранняя
остановка по нему. Полные числа — `metrics.csv`, `summary.json`; PNG — `png/`.""")
    t_all = time.time()
    tc = training.TrainConfig(manifest=cfg.manifest, limit=cfg.limit, val_size=cfg.val_size)
    df, tr, va = training.load_split(tc)
    ren = rnd.load_renderer(ROOT / cfg.renderer, dev)
    ren.corr = tuple(int(v) for v in cfg.render_corr.split(","))
    mh2f2 = rnd.MH2F2.to(dev)
    t0 = time.time()
    tds = sampler = Yxv = None
    if cfg.dataset:                                   # материализованный датасет (маски O и X + метки)
        tds = loader.ObliqueShardDataset(ROOT / cfg.dataset, "train", cache=1)
        vds = loader.ObliqueShardDataset(ROOT / cfg.dataset, "val", cache=1)
        pick = np.unique(np.linspace(0, len(vds) - 1, min(cfg.val_size, len(vds))).round().astype(int))
        pick = pick[np.argsort([vds.shard_of(int(i)) for i in pick], kind="stable")]      # по шардам: одна распаковка
        items = [vds[int(i)] for i in pick]
        Yv = torch.stack([a for a, _, _ in items]); Yxv = torch.stack([b for _, b, _ in items])
        L = torch.stack([c for _, _, c in items]); M1, M2, Dv = L[:, 0], L[:, 4], L[:, 7]
        src = f"{len(tds)} масок (шарды {cfg.dataset})"
    else:
        Yv, Dv, M1, M2 = decode_oblique(va, cfg.component, cfg.workers, seed=cfg.seed + 7)
        src = f"{len(tr)} SAO (синтез на лету)"
    torch.manual_seed(cfg.seed + 1)
    Xv = torch.cat([render_input(ren, mh2f2, Yv[k:k + 128].to(dev).long(),
                                 Yxv[k:k + 128].to(dev).long() if Yxv is not None else None, cfg.input_mode, cfg.bg_shift, cfg.cover, _dens(cfg)).cpu()
                    for k in range(0, len(Yv), 128)]) if len(Yv) else torch.zeros(0)
    print(f"[{cfg.stage}/{cfg.run}] train {src}, val {len(Yv)} (D: {np.unique(Dv.numpy()).tolist()}), "
          f"вход {cfg.input_mode}, рендер val за {time.time() - t0:.0f} с; устройство {dev}", flush=True)
    if tds is not None:
        sampler = loader.ShardSampler(tds, seed=cfg.seed)
        dl = DataLoader(tds, batch_size=cfg.batch, sampler=sampler, num_workers=cfg.workers, drop_last=True,
                        persistent_workers=cfg.workers > 0, prefetch_factor=4 if cfg.workers > 0 else None)
    else:
        dl = DataLoader(loader.ObliqueDataset(tr, component=cfg.component, seed=cfg.seed), batch_size=cfg.batch, shuffle=True,
                        num_workers=cfg.workers, persistent_workers=cfg.workers > 0, prefetch_factor=4 if cfg.workers > 0 else None)
    lset = training.select_logging_set(df[df.split == "val"], ROOT / "runs" / "E1" / "logging_set.json") \
        if (ROOT / "runs" / "E1" / "logging_set.json").exists() else \
        training.select_logging_set(df[df.split == "val"], ROOT / "runs" / cfg.stage / "logging_set.json")
    day_sets = []
    for d in lset["days"]:
        rows = df[(df.station == d["station"]) & (pd.to_datetime(df.time).dt.date.astype(str) == d["date"])].sort_values("time").reset_index(drop=True)
        Yd, _, m1d, m2d = decode_oblique(rows, "O", min(cfg.workers, 2), fixed_d=cfg.track_d)
        Yxd = decode_oblique(rows, "X", min(cfg.workers, 2), fixed_d=cfg.track_d)[0] if cfg.input_mode == "ox" else None
        day_sets.append((d, rows, Yd, Yxd, m1d.numpy(), m2d.numpy()))
    Yi, Di = Yv[:cfg.log_images], Dv[:cfg.log_images]
    print(f"  панелей {len(Yi)}, суток-треков {len(day_sets)} при D={cfg.track_d:.0f}", flush=True)

    net = UNet(2, len(obs.OB_CLASSES), base=cfg.base, depth=cfg.depth).to(dev)
    vz_net = None
    if cfg.vz_weights:
        ck = torch.load(ROOT / cfg.vz_weights, map_location=dev); c = ck["cfg"]
        vz_net = UNet(2, len(canon.CLASSES), base=c["base"], depth=c["depth"], profile=c.get("profile", False)).to(dev)
        vz_net.load_state_dict(ck["state_dict"])
    opt = torch.optim.Adam(net.parameters(), cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs), eta_min=cfg.lr / 100) if cfg.sched == "cosine" else None
    scaler = torch.amp.GradScaler(enabled=(cfg.amp and dev.type == "cuda"))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(CE_WEIGHTS, device=dev))
    vocab = vd.load_vocabulary(); ref_gate: dict = {}
    variant = VARIANTS[cfg.variant]
    print(f"  U-Net {n_params(net)} параметров, вариант {cfg.variant}, рендерер {cfg.renderer}", flush=True)
    best = dict(value=float("inf"), epoch=-1); hist = []
    for ep in range(cfg.epochs):
        net.train(); t_ep = time.time(); n_seen = 0; sums = {"CE": 0.0, "logic": 0.0}
        if sampler is not None:
            sampler.set_epoch(ep)
        for step, batch in enumerate(dl):
            if (cfg.max_steps and step >= cfg.max_steps) or (cfg.steps_per_epoch and step >= cfg.steps_per_epoch):
                break
            y = batch[0].to(dev).long()
            yx = batch[1].to(dev).long() if tds is not None else None
            x = render_input(ren, mh2f2, y, yx, cfg.input_mode, cfg.bg_shift, cfg.cover, _dens(cfg))
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                lg = net(x); loss_ce = ce(lg, y)
            loss = loss_ce
            if variant:
                l_logic, _ = logic.oblique_logic(lg.float(), variant)
                loss = loss + cfg.lam * l_logic; sums["logic"] += l_logic.item() * len(y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sums["CE"] += loss_ce.item() * len(y); n_seen += len(y)
            if cfg.dry:
                print(f"  dry: x {tuple(x.shape)} y {tuple(y.shape)} logits {tuple(lg.shape)} CE {loss_ce.item():.3f} loss {loss.item():.3f}", flush=True)
        if sched is not None:
            sched.step()
        t_train = time.time() - t_ep
        m = {"epoch": ep, "train/CE": sums["CE"] / max(n_seen, 1), "time/train_s": t_train,
             "time/train_samples_per_s": n_seen / max(t_train, 1e-9), "train/lr": opt.param_groups[0]["lr"]}
        if variant:
            m["train/logic"] = sums["logic"] / max(n_seen, 1)
        t_ev = time.time()
        mv, pm, rt = evaluate(net, Xv, Yv, Dv, M1, M2, dev, cfg, vocab, ep, log, ref_gate, vz_net)
        m.update(mv)
        if cfg.tromso_n:                                  # реальные НЗ Тромсё без меток — критерий переноса
            m.update(tromso_probe(net, dev, cfg, vocab, do_gate=(ep % cfg.gate_every == 0 or ep == cfg.epochs - 1)))
        if len(Yi) and (ep % cfg.images_every == 0 or ep == cfg.epochs - 1):
            (rundir / "png").mkdir(exist_ok=True)
            log.ionograms("images/fixed_set", Xv[:len(Yi)].numpy(), Yi.numpy(), pm[:len(Yi)], ep, extent=EXTENT,
                          titles=[f"D={float(d):.0f}" for d in Di], xlabel="МГц", ylabel="P′, км",
                          save=rundir / "png" / f"ionograms_ep{ep:02d}.png")
        for d, rows, Yd, Yxd, m1d, m2d in day_sets:
            if not len(Yd):
                continue
            with torch.no_grad():
                Xd = torch.cat([render_input(ren, mh2f2, Yd[k:k + 128].to(dev).long(),
                                             Yxd[k:k + 128].to(dev).long() if Yxd is not None else None, cfg.input_mode, cfg.bg_shift, cfg.cover, _dens(cfg)).cpu()
                                for k in range(0, len(Yd), 128)])
            pmd, _ = predict(net, Xd, dev)
            f1, f2, _ = muf_readouts(pmd)
            key = f"{d['station']}_{d['date']}_{d['kind']}"
            hours = pd.to_datetime(rows.time).dt.hour.values + pd.to_datetime(rows.time).dt.minute.values / 60
            log.tracks_grid(f"track/{key}", hours, {"MUF1F2": (m1d, f1, "МГц"), "MUF2F2": (m2d, f2, "МГц")}, ep,
                            title=f"{key} D={cfg.track_d:.0f} км", save=rundir / "png" / f"track_{key}_ep{ep:02d}.png")
            for name, pred, lab in (("MUF1F2", f1, m1d), ("MUF2F2", f2, m2d)):
                st = training.err_stats(pred, lab); m[f"track/{key}/{name}_rmse"] = st.get("rmse", np.nan)
        m["time/eval_s"] = time.time() - t_ev
        log.scalars({k: v for k, v in m.items() if k != "epoch"}, ep); log.row(m, ep); hist.append(m)
        ckpt = {"state_dict": net.state_dict(), "cfg": asdict(cfg), "epoch": ep, "metrics": m}
        torch.save(ckpt, rundir / "weights_last.pt")
        crit = m.get("val/MUF1F2_med", np.nan)
        if np.isfinite(crit) and crit < best["value"]:
            best = dict(value=float(crit), epoch=ep); torch.save(ckpt, rundir / "weights.pt")
        print(f"  ep{ep}: train CE {m['train/CE']:.3f}" + (f" logic {m['train/logic']:.4f}" if variant else "")
              + f" | L_hinge {m['val/logic_total']:.4f} IoU F2 {m.get('val/IoU_F2', np.nan):.3f} MH {m.get('val/IoU_MH', np.nan):.3f} "
              f"МПЧ1F2 RMSE {m.get('val/MUF1F2_rmse', np.nan):.2f} med {m.get('val/MUF1F2_med', np.nan):.2f} "
              + (f"| РЕАЛ F2 {100 * m['real/has_F2_frac']:.0f}% МПЧ {m.get('real/muf1F2_med', np.nan):.2f}±{m.get('real/muf1F2_iqr', np.nan):.2f} "
                 f"гейт {100 * m.get('real/gate_violations', np.nan):.0f}% " if "real/has_F2_frac" in m else "")
              + f"МПЧ2F2 RMSE {m.get('val/MUF2F2_rmse', np.nan):.2f} inv {m.get('val/inv_ratio_pred_med', np.nan):.3f}/{m.get('val/inv_ratio_label_med', np.nan):.3f} "
              f"gate {m.get('gate/violations', np.nan):.0%} (метки {m.get('gate/labels_violations', np.nan):.0%}) | {t_train:.0f}+{m['time/eval_s']:.0f} с, "
              f"{m['time/train_samples_per_s']:.0f} обр/с", flush=True)
    rt.to_csv(rundir / "val_readouts.csv", index=False)
    summary = {"run": cfg.run, "stage": cfg.stage, "variant": cfg.variant, "params": n_params(net), "n_val": int(len(Yv)),
               "best_epoch": best["epoch"], "best_value": best["value"], "best": hist[best["epoch"]] if best["epoch"] >= 0 else {},
               "last": hist[-1], "time_total_s": time.time() - t_all}
    (rundir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log.close()
    print(f"[{cfg.stage}/{cfg.run}] готово за {time.time() - t_all:.0f} с → {rundir}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(ObliqueConfig):
        if f.type is bool or f.type == "bool":
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    train_oblique(ObliqueConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
