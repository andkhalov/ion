# -*- coding: utf-8 -*-
"""
anchor.py — привязка реальных НЗ-снимков к НЕЗАВИСИМОМУ вертикальному зонду (Э4 §5.6).

Зачем: беcметочный критерий отбирает чекпойнт, но сам ни с чем не сверен, и на грубом, локально
гладком завышении МПЧ он слеп (Э4 §5.7). Внешний арбитр — вертикальный ионозонд в пункте трассы.

Цепочка: характеристики ARTIST вертикального зонда (GIRO fastchar, без логина) → параболический
слой F2 по (foF2, hmF2, yF2) → аналитический след h′(f) = h₀ + (y/2)(f/fo)·ln((fo+f)/(fo−f)),
h₀ = hm − y → `oblique_synth.muf` (сферический секанс) → ожидаемая МПЧ на дальность трассы.
Сравниваем с тем, что модель читает с реального снимка чирп-зонда (или с независимым отсчётом
`nose_readout`, который модель не использует — им проверяется сам закон секанса).

Оговорки, обязательные рядом с числами: зонд стоит в ПУНКТЕ трассы, отражение — в середине;
параболический слой — модель, а не измеренный след; для SGO→TGO доступен только приёмный конец.

  python -m pyon.anchor --fetch                       — догрузить характеристики GIRO в data/giro/
  python -m pyon.anchor --route sgo-tgo --weights runs/E5/baseline/weights.pt   — сверка + трек
  python -m pyon.anchor --route sgo-tgo --nose        — независимый отсчёт носа (проверка секанса)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import oblique_synth as obs                                       # noqa: E402

GIRO = "https://lgdc.uml.edu/fastchar/getbest"
CHARS = ("foF2", "hmF2", "yF2")
ROUTES = {"sgo-tgo": dict(d_km=430.0, station="TR169"),                    # Соданкюля → Тромсё
          "juliusruh-tgo": dict(d_km=1500.0, station="JR055")}             # Юлиусру → Тромсё
DATA = ROOT / "data" / "giro"


def fetch(station: str, char: str, day_from: str, day_to: str) -> Path:
    """Скачать характеристику станции за интервал (idempotent: готовый файл не перекачиваем)."""
    import urllib.parse, urllib.request
    DATA.mkdir(parents=True, exist_ok=True)
    tag = f"{day_from}_{day_to}".replace("/", "").replace(".", "").replace(":", "")   # дата содержит "/"
    out = DATA / f"{station}_{char}_{tag}.txt"
    if out.exists() and out.stat().st_size > 400:
        return out
    q = urllib.parse.urlencode(dict(ursiCode=station, charName=char, fromDate=day_from, toDate=day_to))
    with urllib.request.urlopen(f"{GIRO}?{q}", timeout=120) as r:
        out.write_bytes(r.read())
    return out


def read_char(path: Path, char: str) -> pd.DataFrame:
    """Файл fastchar → (время, оценка достоверности ARTIST, значение); строки-заголовки пропускаем."""
    rows = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        try:
            rows.append((pd.Timestamp(p[0]), float(p[1]), float(p[2])))
        except (ValueError, IndexError):
            continue
    d = pd.DataFrame(rows, columns=["t", "cs", char]).dropna()
    return d[d.cs >= 0].sort_values("t")


def parabolic_trace(fo: float, hm: float, y: float, n: int = 400):
    """След h′(f) параболического слоя (обыкновенная волна) — стандартный результат для
    параболы: h′ = h₀ + (y/2)·(f/fo)·ln((fo+f)/(fo−f)), h₀ = hm − y."""
    f = np.linspace(0.05 * fo, 0.995 * fo, n)
    h = (hm - y) + 0.5 * y * (f / fo) * np.log((fo + f) / (fo - f))
    ok = np.isfinite(h) & (h > 0) & (h < 1500)
    return f[ok], h[ok]


def expected(route: str, day_from: str, day_to: str, hops: int = 1) -> pd.DataFrame:
    """Ожидаемая МПЧ трассы по характеристикам вертикального зонда в её пункте."""
    st, d_km = ROUTES[route]["station"], ROUTES[route]["d_km"]
    d = read_char(fetch(st, "foF2", day_from, day_to), "foF2")
    for ch in CHARS[1:]:
        d = pd.merge_asof(d, read_char(fetch(st, ch, day_from, day_to), ch)[["t", ch]],
                          on="t", tolerance=pd.Timedelta("3min"), direction="nearest")
    d = d.dropna(subset=list(CHARS))
    out = []
    for r in d.itertuples():
        fv, hv = parabolic_trace(r.foF2, r.hmF2, r.yF2)
        if len(fv) >= 10:
            out.append((r.t, r.foF2, r.hmF2, r.yF2, obs.muf(fv, hv, d_km, hops, "spherical")))
    return pd.DataFrame(out, columns=["t", "foF2", "hmF2", "yF2", "muf_exp"])


def nose_readout(x: np.ndarray, cov: np.ndarray, min_span: int = 6) -> float:
    """НЕЗАВИСИМЫЙ от модели отсчёт МПЧ — им проверяется сам закон секанса, а не чтение модели:
    берём связную компоненту активных пикселей с наибольшим размахом по частоте и возвращаем её
    верхнюю частоту. Колоночный вариант (просто «последняя колонка с откликом») ОТВЕРГНУТ
    2026-09-08: на реальных снимках Тромсё он завышал МПЧ на 8.7 МГц систематически, потому что
    брал помеховые полосы на высоких частотах, а не след."""
    from scipy import ndimage
    act = (((x > 0).any(0) if x.ndim == 3 else (x > 0)) & cov)
    lab, n = ndimage.label(act, structure=np.ones((3, 3), int))
    best, span = -1, 0
    for k in range(1, n + 1):
        cols = np.flatnonzero((lab == k).any(0))
        if len(cols) and (cols[-1] - cols[0]) > span:
            span, best = cols[-1] - cols[0], cols[-1]
    return float(obs.fob_axis[best]) if span >= min_span else float("nan")


def track_figure(m: pd.DataFrame, route: str, out: Path, title: str = ""):
    """Суточный трек: МПЧ модели против МПЧ по прибору + foF2 зонда для контекста."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, height_ratios=[2, 1], constrained_layout=True)
    ax[0].plot(m.t, m.muf_exp, "-", lw=2, color="tab:blue", label=f"по вертикальному зонду {ROUTES[route]['station']}")
    ok = m.muf_model.notna()
    ax[0].plot(m.t[ok], m.muf_model[ok], "o", ms=4, color="tab:red", label="наша модель по снимку чирп-зонда")
    bad = ok & ((m.muf_model - m.muf_exp).abs() > 2)
    ax[0].plot(m.t[bad], m.muf_model[bad], "o", ms=9, mfc="none", mec="black", mew=1.2,
               label=f"грубые промахи > 2 МГц ({int(bad.sum())})")
    ax[0].set_ylabel("МПЧ 1F2, МГц"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=9)
    ax[0].set_title(title or f"{route}, D = {ROUTES[route]['d_km']:.0f} км")
    ax[1].plot(m.t, m.foF2, "-", color="tab:green", label="foF2 зонда")
    ax[1].set_ylabel("foF2, МГц"); ax[1].set_xlabel("UTC"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H"))
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def compare(route: str, weights: str = "", nose: bool = False, x_mode: str = "zero", active: float = 0.15):
    """Сопоставление по времени: модель и/или независимый отсчёт против прибора."""
    from pyon import tromso as tg
    d = ROOT / "data" / "tromso" / route
    files = sorted(d.glob("*.png"))
    xs, cov, ts = [], [], []
    for fp in files:
        try:
            snr, f, r = tg.png_to_snr(fp, route)
            x, cv = tg.rasterize(snr, f, r, x_mode, active, None)
            xs.append(x); cov.append(cv); ts.append(pd.Timestamp(fp.stem))
        except Exception:
            continue
    X, cov, ts = np.stack(xs), np.stack(cov), pd.to_datetime(ts)
    got = pd.DataFrame({"t": ts}).sort_values("t").reset_index(drop=True)
    order = np.argsort(ts.values)
    if weights:
        import torch
        from pyon import oblique_train as OT
        from pyon.models import UNet
        ck = torch.load(ROOT / weights, map_location="cuda", weights_only=False); c = ck["cfg"]
        net = UNet(2, len(obs.OB_CLASSES), base=c["base"], depth=c["depth"]).to("cuda")
        net.load_state_dict(ck["state_dict"]); net.eval()
        pm, _ = OT.predict(net, torch.from_numpy(X).float().div(255), "cuda")
        pm = np.where(cov, pm, 0).astype(pm.dtype)
        got["muf_model"] = OT.muf_readouts(pm)[0][order]
    if nose:
        got["muf_nose"] = [nose_readout(X[i], cov[i]) for i in order]
    day = (ts.min().strftime("%Y/%m/%d"), (ts.max() + pd.Timedelta(days=1)).strftime("%Y/%m/%d"))
    exp = expected(route, *day)
    m = pd.merge_asof(got, exp, on="t", tolerance=pd.Timedelta("6min"), direction="nearest")
    return m.dropna(subset=["muf_exp"])


def stats(m: pd.DataFrame, col: str) -> dict:
    ok = m.dropna(subset=[col])
    dd = ok[col] - ok.muf_exp
    within = (dd.abs() <= 1.0)
    return dict(n=len(ok), n_all=len(m), frac=len(ok) / max(len(m), 1), med=float(dd.abs().median()),
                rmse=float(np.sqrt((dd ** 2).mean())), bias=float(dd.median()),
                corr=float(ok[col].corr(ok.muf_exp)), within1=float(within.mean()),
                corr1=float(ok[col][within].corr(ok.muf_exp[within])), gross=int((dd.abs() > 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="sgo-tgo", choices=list(ROUTES))
    ap.add_argument("--weights", default="")
    ap.add_argument("--nose", action="store_true")
    ap.add_argument("--fig", default="")
    a = ap.parse_args()
    m = compare(a.route, a.weights, a.nose)
    for col, name in (("muf_model", "модель"), ("muf_nose", "независимый отсчёт носа")):
        if col in m:
            s = stats(m, col)
            print(f"{name}: найдено {s['n']}/{s['n_all']} ({100*s['frac']:.0f} %) | |Δ| мед {s['med']:.2f} "
                  f"RMSE {s['rmse']:.2f} смещение {s['bias']:+.2f} | корреляция {s['corr']:.3f} | "
                  f"в пределах 1 МГц {100*s['within1']:.0f} % (корр {s['corr1']:.3f}) | промахов > 2 МГц {s['gross']}")
    if a.fig and "muf_model" in m:
        print("рисунок →", track_figure(m, a.route, ROOT / a.fig))
    m.to_csv(ROOT / f"local/anchor_{a.route}.csv", index=False)


if __name__ == "__main__":
    main()
