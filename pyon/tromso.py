# -*- coding: utf-8 -*-
"""
tromso.py — E6: реальные НЗ-ионограммы Тромсё (chirpsounder2, J. Vierinen; дашборд отдаёт PNG) →
обратная растеризация в НЗ-решётку `oblique_synth` (2–24 МГц × 300–3200 км) → инференс НЗ-модели
E5 → ридауты (МПЧ 1F2/2F2, P′ носа) → SHACL-гейт на реальных сценах → суточный ход МПЧ и панели
(Э3 §4 п.4; сравнение с ручной разметкой `data/manual/oblique_labels.csv` — отдельным шагом).

Геометрия картинки (1200×900, matplotlib; проверено 2026-09-06): бокс осей x 79..964, y 58..842;
шкалы линейные — частота 0.5–16 МГц, «one-way range offset» 200–1500 км (для трассы SGO→TGO 430 км
след F2 на 600–800 км = групповой путь P′ = √(D² + (2h′)²) — т.е. ось — групповой путь одного
скачка, берём как P′); яркость — SNR 0 (белый) … 20 дБ (чёрный). Бокс ищется автоматически по
спайнам (длинные тёмные линии), пределы осей заданы константами AXES.

Нормировка — ПОД ДОМЕН ОБУЧЕНИЯ (реальное сырьё дигизонда после `read_canon`: ~15 % активных
пикселей, амплитуды активных до 255): шкала SNR чирп-зонда (0–20 дБ) не совпадает с 24-дБ шкалой
дигизонда, а порог «медиана + 6 дБ» оставлял 0.8 % активных и тусклые следы (коды ≤ 100) — модель
E5 их не видела (проверка 2026-09-06). Поэтому: порог = квантиль SNR (1 − active), сигнал =
clip(SNR − порог) растянут так, что 99.9-й перцентиль → 255; в ячейку решётки — максимум по
попавшим пикселям; 3-px отступ от рамки осей (спайн давал ложную линию на 15.5 МГц). Канал X: у
чирп-зонда нет разделения поляризаций — `x_mode` zero (нули) или copy (копия O); что ближе к
обучению — решает сравнение гейта/ридаутов (DIARY).

Запуск: python -m pyon.tromso --weights runs/E5/lognorm/weights.pt --route sgo-tgo [--x_mode zero]
Артефакты: runs/E6/<ран>/<route>/{readouts.csv, metrics.json, png/panel_*.png, png/muf_course.png}.
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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import canon, gates, tblog                                             # noqa: E402
from pyon import oblique_synth as obs                                            # noqa: E402
from pyon import validate as vd                                                  # noqa: E402
from pyon.oblique_train import muf_readouts, predict                              # noqa: E402
from pyon.models import UNet                                                     # noqa: E402

AXES = dict(f=(0.5, 16.0), r=(200.0, 1500.0), snr_max_db=20.0)                   # пределы осей дашборда
ROUTES = {"sgo-tgo": 430.0, "juliusruh-tgo": 1500.0}                             # длина трассы, км (D)


def axes_box(gray: np.ndarray):
    """Бокс осей matplotlib: длинные тёмные линии (спайны); колорбар справа исключаем."""
    dark = gray < 60
    rows = np.flatnonzero(dark.mean(1) > 0.6); cols = np.flatnonzero(dark.mean(0) > 0.6)
    cols = cols[cols < gray.shape[1] * 0.85]
    if len(rows) < 2 or len(cols) < 2:
        raise ValueError("не найден бокс осей")
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())


def png_to_snr(path: str | Path):
    """PNG → (SNR [h, w] дБ внутри бокса осей, оси f [w] МГц, r [h] км)."""
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    x0, x1, y0, y1 = axes_box(gray)
    m = 3                                                                # отступ от спайнов (антиалиасинг)
    inner = gray[y0 + m:y1 - m + 1, x0 + m:x1 - m + 1]
    snr = AXES["snr_max_db"] * (1.0 - inner / 255.0)
    fa, fb = AXES["f"]; ra, rb = AXES["r"]
    W, H = x1 - x0, y1 - y0
    f = fa + (fb - fa) * (np.arange(inner.shape[1]) + m) / W
    r = rb - (rb - ra) * (np.arange(inner.shape[0]) + m) / H
    return snr, f, r


def rasterize(snr: np.ndarray, f: np.ndarray, r: np.ndarray, x_mode: str = "zero", active: float = 0.15,
              pad: np.ndarray | None = None):
    """SNR-картинка → (X uint8 [2, NP, NF] на решётке oblique_synth, covered [NP, NF] — где есть данные):
    порог = квантиль (1 − active) SNR, растяжение так, что 99.9-й перцентиль сигнала → 255, max по ячейке;
    вне поля картинки — pad (фон рендерера E4 [2, NP, NF] 0..255) или нули."""
    keep = f <= AXES["f"][1] - 1.0                                   # крайние бины по частоте — яркая кромка картинки (до ~15.3 МГц)
    snr, f = snr[:, keep], f[keep]
    jf = canon.to_grid(f, obs.FOB_MIN, obs.FOB_MAX, obs.NF); jp = canon.to_grid(r, obs.P_MIN, obs.P_MAX, obs.NP)
    okf, okp = jf >= 0, jp >= 0
    # сначала укрупнение (max SNR по ~6×6 пикселям ячейки), затем порог по квантилю УКРУПНЁННОЙ карты —
    # иначе после max-пулинга активны почти все ячейки (88 % при квантиле пикселей 0.85; 2026-09-06)
    cell = np.full((obs.NP, obs.NF), -np.inf, np.float32)
    sub = snr[np.ix_(okp, okf)]
    np.maximum.at(cell, (jp[okp][:, None].repeat(okf.sum(), 1).ravel(), jf[okf][None, :].repeat(okp.sum(), 0).ravel()), sub.ravel())
    covered0 = np.isfinite(cell)
    vals = cell[covered0]
    thr = float(np.quantile(vals, 1.0 - active))
    sig = np.where(covered0, np.clip(cell - thr, 0, None), 0.0)
    top = float(np.quantile(sig[sig > 0], 0.999)) if (sig > 0).any() else 1.0
    out = np.clip(sig / max(top, 1e-6), 0, 1) * 255.0
    # вне поля картинки (f > 16 МГц, P′ > 1500 км): резкая граница нулей читалась моделью как след/
    # кратник, iid-подложка из пикселей фона — как ложные следы (МПЧ → 23 МГц); поэтому подкладываем
    # фон РЕНДЕРЕРА E4 (пустая маска → коррелированный спекл, на котором модель училась) — 2026-09-06
    covered = np.zeros((obs.NP, obs.NF), bool); covered[np.ix_(np.unique(jp[okp]), np.unique(jf[okf]))] = True
    o = out.round().astype(np.uint8)
    x = np.stack([o, o if x_mode == "copy" else np.zeros_like(o)])
    if pad is not None:
        # подложку приводим к статистике внутренней области: та же доля активных и та же средняя амплитуда
        pad = pad.astype(np.float32)
        act_in = float((o[covered] > 0).mean()); amp_in = float(o[covered][o[covered] > 0].mean()) if (o[covered] > 0).any() else 0.0
        q = np.quantile(pad[0], 1.0 - act_in) if act_in > 0 else np.inf
        padm = np.where(pad > q, pad, 0.0)
        amp_pad = float(padm[padm > 0].mean()) if (padm > 0).any() else 1.0
        padm = np.clip(padm * (amp_in / max(amp_pad, 1e-6)), 0, 255)
        x = np.where(covered[None], x, padm.round().astype(np.uint8))
    return x, covered


def load_ob_net(weights: Path, dev):
    ck = torch.load(weights, map_location=dev); c = ck["cfg"]
    net = UNet(2, len(obs.OB_CLASSES), base=c["base"], depth=c["depth"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--route", default="sgo-tgo", choices=list(ROUTES))
    ap.add_argument("--x_mode", default="zero", choices=["zero", "copy"])
    ap.add_argument("--run", default="")
    ap.add_argument("--gate_n", type=int, default=150)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--panels", type=int, default=12)
    ap.add_argument("--active", type=float, default=0.15, help="целевая доля активных пикселей (домен обучения ~0.15)")
    ap.add_argument("--renderer", default="runs/E4/hetero/weights.pt", help="фон рендерера вне поля картинки ('' — нули)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = a.run or f"{Path(a.weights).parent.name}_{a.x_mode}_a{int(round(100 * a.active))}"
    rundir = ROOT / "runs" / "E6" / run / a.route; (rundir / "png").mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "data" / "tromso" / a.route).glob("*.png"))
    if not files:
        print("нет картинок:", ROOT / "data" / "tromso" / a.route); return
    net, c = load_ob_net(ROOT / a.weights, dev)
    pad = None
    if a.renderer:
        from pyon import renderer as rnd
        ren = rnd.load_renderer(ROOT / a.renderer, dev); torch.manual_seed(0)
        pad = (ren.sample(torch.zeros(1, obs.NP, obs.NF, dtype=torch.long, device=dev))[0] * 255).round().cpu().numpy()
    t0 = time.time()
    xs, cov, times = [], [], []
    for fp in files:
        try:
            snr, f, r = png_to_snr(fp); x, cv = rasterize(snr, f, r, a.x_mode, a.active, pad); xs.append(x); cov.append(cv); times.append(fp.stem)
        except Exception as e:                                   # битая/неполная картинка
            print("  пропуск", fp.name, e)
    X = torch.from_numpy(np.stack(xs)).float().div(255)
    pm, comps = predict(net, X, dev)
    from scipy.ndimage import binary_erosion
    cov_e = np.stack([binary_erosion(c_, iterations=2) for c_ in cov])          # граница поля (2 ячейки) — тоже фон
    pm = np.where(cov_e, pm, 0).astype(pm.dtype)                                  # вне поля данных — фон
    f1, f2, pn = muf_readouts(pm)
    t = pd.DataFrame(dict(time=pd.to_datetime(times, format="%Y%m%dT%H%M%SZ", utc=True), muf1F2=f1, muf2F2=f2, p_nose=pn,
                          logic=sum(comps.values()) if comps else np.nan,
                          has_F2=[(p == obs.OB_CLASSES.index("F2")).sum() >= 4 for p in pm],
                          has_MH=[(p == obs.OB_CLASSES.index("MH")).sum() >= 4 for p in pm],
                          active_frac=[(x[0][c_] > 0).mean() for x, c_ in zip(xs, cov)]))
    t["ratio_2F2_1F2"] = t.muf2F2 / t.muf1F2
    m = dict(n=len(t), route=a.route, D_km=ROUTES[a.route], x_mode=a.x_mode,
             has_F2_frac=float(t.has_F2.mean()), has_MH_frac=float(t.has_MH.mean()),
             muf1F2_median=float(np.nanmedian(t.muf1F2)), muf1F2_iqr=float(np.nanpercentile(t.muf1F2, 75) - np.nanpercentile(t.muf1F2, 25)) if t.muf1F2.notna().any() else np.nan,
             ratio_median=float(np.nanmedian(t.ratio_2F2_1F2)), ratio_iqr=float(np.nanpercentile(t.ratio_2F2_1F2.dropna(), 75) - np.nanpercentile(t.ratio_2F2_1F2.dropna(), 25)) if t.ratio_2F2_1F2.notna().any() else np.nan,
             active_frac_median=float(t.active_frac.median()))
    n_g = min(a.gate_n, len(pm))
    if n_g:
        vocab = vd.load_vocabulary()
        rate, warn, _ = gates.gate_rate(pm[:n_g], gates.oblique_scene, vocab, prefix="tgo_", procs=a.procs, with_warnings=True)
        m.update(gate_violations=rate, gate_warnings=warn, gate_n=n_g)
    log = tblog.TBLog(rundir, dict(weights=a.weights, route=a.route, x_mode=a.x_mode))
    ext = (obs.FOB_MIN, obs.FOB_MAX, obs.P_MIN, obs.P_MAX)
    ix = np.unique(np.linspace(0, len(xs) - 1, min(a.panels, len(xs))).round().astype(int))
    log.ionograms("E6/panels", X[ix].numpy(), np.zeros((len(ix), obs.NP, obs.NF), np.int8), pm[ix], 0, extent=ext,
                  titles=[f"{a.route} {times[i]}" for i in ix], n_classes=len(obs.OB_CLASSES), xlabel="МГц", ylabel="P′, км",
                  save=rundir / "png" / "panels.png")
    if len(t) > 1:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(t.time, t.muf1F2, "o-", ms=3, label="МПЧ 1F2 (модель)"); ax.plot(t.time, t.muf2F2, "s--", ms=3, label="МПЧ 2F2 (модель)")
        ax.set_ylabel("МГц"); ax.set_title(f"{a.route} (D={ROUTES[a.route]:.0f} км): суточный ход МПЧ по реальным НЗ, n={len(t)}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(rundir / "png" / "muf_course.png", dpi=90); log.w.add_figure("E6/muf_course", fig, 0); plt.close(fig)
    t.to_csv(rundir / "readouts.csv", index=False)
    log.scalars({f"E6/{k}": v for k, v in m.items() if isinstance(v, (int, float))}, 0)
    json.dump(m, open(rundir / "metrics.json", "w"), indent=1, ensure_ascii=False)
    print(f"[E6/{run}/{a.route}] {len(t)} ионограмм за {time.time() - t0:.0f} с: F2 найден у {m['has_F2_frac']:.0%}, MH у {m['has_MH_frac']:.0%}; "
          f"МПЧ1F2 медиана {m['muf1F2_median']:.2f} МГц (IQR {m['muf1F2_iqr']:.2f}); инвариант 2F2/1F2 {m['ratio_median']:.3f}; "
          f"гейт {100 * m.get('gate_violations', np.nan):.1f} % (предупр. {100 * m.get('gate_warnings', np.nan):.1f} %) → {rundir}", flush=True)


if __name__ == "__main__":
    main()
