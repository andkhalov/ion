"""
Минимальные независимые парсеры форматов Digisonde (DPS-4 / DPS-4D):

  RSF / SBF  — «сырые» ионограммы (бинарные блоки по 4096 байт)
  SAO 4.x    — результат автоскейлинга ARTIST (текст, фиксированные форматы)
  EDP        — профиль электронной концентрации (текст, таблица)
  DFT        — доплеровские спектры дрейфовых измерений (бинарный)

Написаны по Digisonde-4D System Manual, Annex 5C (таблицы 5C-37 … 5C-50)
и SAO-4.3 (http://ulcar.uml.edu/~iag/SAO-4.3.htm). Используются для
самостоятельной проверки данных и как наивная реализация рядом с pynasonde.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK = 4096
HEADER_LEN = 60
PRELUDE_LEN = 6

# Таблицы 5C-39 (RSF, 2 байта/бин) и 5C-44 (SBF, 1 байт/бин)
RSF_LAYOUT = {128: (15, 128), 256: (8, 249), 512: (4, 501)}   # (groups per block, bins per group)
SBF_LAYOUT = {128: (30, 128), 256: (15, 256), 512: (8, 498)}

OFFSET_CODES = {0: -20e3, 1: -10e3, 2: 0.0, 3: 10e3, 4: 20e3, 5: np.nan, 0xE: np.nan, 0xF: np.nan}
OFFSET_LABELS = {5: "search failure", 0xE: "forced", 0xF: "no Tx"}


def bcd(b: int) -> int:
    return 10 * (b >> 4) + (b & 0xF)


@dataclass
class Preface:
    """Общий PREFACE (таблица 5C-37) — настройки зондирования."""
    record_type: int
    header_length: int
    version: int
    date: dt.datetime
    doy: int
    stn_rx: str
    stn_tx: str
    schedule: int
    program: int
    f_start_hz: float
    f_coarse_step_hz: float
    f_stop_hz: float
    f_fine_step_hz: float
    n_small_steps: int
    phase_code: int
    antenna_option: int
    n_samples_log2: int
    prr_pps: int
    range_start_km: float
    range_inc_km: float
    n_heights: int
    delay: int
    base_gain: int
    freq_search: int
    operating_mode: int
    data_format: int
    printer: int
    threshold: int
    constant_gain: int
    cit_length_ms: int
    journal: int
    bottom_window_km: int
    top_window_km: int
    n_heights_stored: int

    @property
    def both_polarizations(self) -> bool:
        return self.antenna_option < 8

    @property
    def n_doppler_lines(self) -> int:
        return 2 ** self.n_samples_log2

    @property
    def format_name(self) -> str:
        return {1: "MMM", 2: "DFT", 3: "PGH", 4: "RSF", 5: "SBF", 6: "BIT"}.get(self.data_format, "?")


def parse_header(h: bytes) -> Preface:
    assert len(h) == HEADER_LEN
    year = 2000 + bcd(h[3])
    doy = bcd(h[4]) * 100 + bcd(h[5])
    month, dom, hour, minute, sec = (bcd(h[i]) for i in range(6, 11))
    inc = {2: 2.5, 5: 5.0, 10: 10.0}.get(bcd(h[37]), float(bcd(h[37])))
    return Preface(
        record_type=h[0], header_length=h[1], version=h[2],
        date=dt.datetime(year, month, dom, hour, minute, sec), doy=doy,
        stn_rx=h[11:14].decode("ascii", "replace"), stn_tx=h[14:17].decode("ascii", "replace"),
        schedule=bcd(h[17]), program=bcd(h[18]),
        f_start_hz=(bcd(h[19]) * 1e4 + bcd(h[20]) * 1e2 + bcd(h[21])) * 100,
        f_coarse_step_hz=(bcd(h[22]) * 100 + bcd(h[23])) * 1e3,
        f_stop_hz=(bcd(h[24]) * 1e4 + bcd(h[25]) * 1e2 + bcd(h[26])) * 100,
        f_fine_step_hz=(bcd(h[27]) * 100 + bcd(h[28])) * 1e3,
        n_small_steps=int(np.int8(h[29])), phase_code=bcd(h[30]), antenna_option=int(np.int8(h[31])),
        n_samples_log2=bcd(h[32]), prr_pps=bcd(h[33]) * 100 + bcd(h[34]),
        range_start_km=bcd(h[35]) * 100 + bcd(h[36]), range_inc_km=inc,
        n_heights=bcd(h[38]) * 100 + bcd(h[39]), delay=bcd(h[40]) * 100 + bcd(h[41]),
        base_gain=bcd(h[42]), freq_search=bcd(h[43]), operating_mode=bcd(h[44]),
        data_format=bcd(h[45]), printer=bcd(h[46]), threshold=bcd(h[47]), constant_gain=bcd(h[48]),
        cit_length_ms=int.from_bytes(h[51:53], "big"), journal=h[53],
        bottom_window_km=bcd(h[54]) * 100 + bcd(h[55]), top_window_km=bcd(h[56]) * 100 + bcd(h[57]),
        n_heights_stored=bcd(h[58]) * 100 + bcd(h[59]),
    )


def read_ionogram(path: str | Path) -> tuple[Preface, pd.DataFrame]:
    """Читает RSF или SBF. Возвращает (PREFACE первого блока, long-таблица эхо-сигналов).

    Колонки таблицы: block, pol, freq_hz, offset_hz, add_gain_db, seconds, mpa_db,
    height_km, amp_db, doppler, (phase_deg, azimuth_deg — только RSF).
    Амплитуда — в 3-дБ единицах (0..31 → 0..93 дБ), MPA (most probable amplitude) в тех же единицах.
    """
    raw = Path(path).read_bytes()
    n_blocks = len(raw) // BLOCK
    rows = []
    preface0 = None
    for b in range(n_blocks):
        blk = raw[b * BLOCK:(b + 1) * BLOCK]
        pf = parse_header(blk[:HEADER_LEN])
        preface0 = preface0 or pf
        is_rsf = pf.record_type in (6, 7)
        layout = RSF_LAYOUT if is_rsf else SBF_LAYOUT
        bytes_per_bin = 2 if is_rsf else 1
        n_groups, n_bins = layout[pf.n_heights]
        gsize = PRELUDE_LEN + n_bins * bytes_per_bin
        pos = HEADER_LEN
        heights = pf.range_start_km + np.arange(n_bins) * pf.range_inc_km
        for _ in range(n_groups):
            pre = blk[pos:pos + PRELUDE_LEN]
            if len(pre) < PRELUDE_LEN or pre[0] == 0xEE or pre == b"\x00" * 6:
                break  # END-OF-IONOGRAM или пустой хвост блока
            pol = {3: "O", 2: "X"}.get(pre[0] >> 4, "?")
            freq_hz = (bcd(pre[1]) * 100 + bcd(pre[2])) * 10e3
            offset_code, add_gain = pre[3] >> 4, pre[3] & 0xF
            seconds, mpa = bcd(pre[4]), bcd(pre[5])
            data = np.frombuffer(blk[pos + PRELUDE_LEN:pos + gsize], dtype=np.uint8)
            if is_rsf:
                b1, b2 = data[0::2], data[1::2]
                amp, dop = b1 >> 3, b1 & 7
                phase, azm = b2 >> 3, b2 & 7
            else:
                amp, dop = data >> 3, data & 7
                phase = azm = None
            rec = dict(block=b, pol=pol, freq_hz=freq_hz, offset_hz=OFFSET_CODES.get(offset_code, np.nan),
                       offset_flag=OFFSET_LABELS.get(offset_code, ""), add_gain_db=3 * add_gain,
                       seconds=seconds, mpa_db=3 * mpa)
            df = pd.DataFrame(dict(height_km=heights, amp_db=3 * amp.astype(float), doppler=dop.astype(int)))
            if is_rsf:
                df["phase_deg"] = 11.25 * phase.astype(float)
                df["azimuth_deg"] = 60 * azm.astype(int)
            for k, v in rec.items():
                df[k] = v
            rows.append(df)
            pos += gsize
    out = pd.concat(rows, ignore_index=True)
    out["freq_mhz"] = out.freq_hz / 1e6
    return preface0, out


def read_canon(path: str | Path, nf: int = 128, nh: int = 128,
               f_min: float = 1.0, f_max: float = 15.0,
               h_min: float = 80.0, h_max: float = 720.0) -> np.ndarray:
    """Быстрый путь для потокового лоадера (Э3 §2.2): RSF|SBF → каноническая матрица
    uint8 [2(O,X), nh, nf] БЕЗ pandas (~10× быстрее read_ionogram+растеризация).

    Семантика согласована с прототипной pyon.dataset_cache.canon_matrix_u8
    (порог = медиана положительных амплитуд поляризации + 6 дБ; MPA из PREFACE у
    DPS-4D в других единицах — не использовать; 0..24 дБ → 0..255) с одним отличием
    В ЛУЧШУЮ сторону: агрегация по высоте — max по нативным бинам ячейки (прототип
    брал ближайший бин и терял эхо); пикселей сигнала получается ~2× больше,
    бинарное совпадение с прототипом ~0.90-0.93. Скорость: 4-15 мс/файл (33-44×).
    """
    raw = Path(path).read_bytes()
    groups = {0: [], 1: []}                     # pol -> [(jf, amp_вектор)]
    jh_cache: dict[tuple, np.ndarray] = {}
    for b in range(len(raw) // BLOCK):
        blk = raw[b * BLOCK:(b + 1) * BLOCK]
        pf = parse_header(blk[:HEADER_LEN])
        is_rsf = pf.record_type in (6, 7)
        layout = RSF_LAYOUT if is_rsf else SBF_LAYOUT
        bpb = 2 if is_rsf else 1
        n_groups, n_bins = layout[pf.n_heights]
        gsize = PRELUDE_LEN + n_bins * bpb
        key = (pf.range_start_km, pf.range_inc_km, n_bins)
        if key not in jh_cache:
            hgt = pf.range_start_km + np.arange(n_bins) * pf.range_inc_km
            jh = np.round((hgt - h_min) / (h_max - h_min) * (nh - 1)).astype(np.int32)
            jh[(hgt < h_min) | (hgt > h_max)] = -1
            jh_cache[key] = jh
        pos = HEADER_LEN
        for _ in range(n_groups):
            pre = blk[pos:pos + PRELUDE_LEN]
            if len(pre) < PRELUDE_LEN or pre[0] == 0xEE or pre == b"\x00" * 6:
                break
            pol = {3: 0, 2: 1}.get(pre[0] >> 4)
            f_mhz = (bcd(pre[1]) * 100 + bcd(pre[2])) * 0.01
            data = np.frombuffer(blk[pos + PRELUDE_LEN:pos + gsize], dtype=np.uint8)
            pos += gsize
            if pol is None:
                continue
            amp = 3.0 * ((data[0::2] if is_rsf else data) >> 3)
            jf = int(round((f_mhz - f_min) / (f_max - f_min) * (nf - 1)))
            groups[pol].append((jf, jh_cache[key], amp))
    out = np.zeros((2, nh, nf), np.uint8)
    for pol, gs in groups.items():
        if not gs:
            continue
        allamp = np.concatenate([a for _, _, a in gs])
        pos_amp = allamp[allamp > 0]
        thr = (np.median(pos_amp) + 6.0) if len(pos_amp) else 0.0
        for jf, jh, amp in gs:
            if jf < 0 or jf >= nf:
                continue
            v = np.clip(amp - thr, 0, 24)
            ok = (jh >= 0) & (v > 0)
            col = out[pol, :, jf]
            np.maximum.at(col, jh[ok], (v[ok] * (255.0 / 24)).astype(np.uint8))
    return out


# ----------------------------------------------------------------------------- SAO 4.x
SAO_GROUP_FORMATS = [  # (name, width, per line, kind) — порядок групп в SAO 4.3 (60 групп)
    ("geophys_const", 7, 16, float), ("system_desc", 120, 1, str), ("time_settings", 120, 1, str),
    ("scaled", 8, 15, float), ("analysis_flags", 2, 60, int), ("doppler_table", 7, 16, float),
    ("F2o_vh", 8, 15, float), ("F2o_th", 8, 15, float), ("F2o_amp", 3, 40, int), ("F2o_dop", 1, 120, int), ("F2o_freq", 8, 15, float),
    ("F1o_vh", 8, 15, float), ("F1o_th", 8, 15, float), ("F1o_amp", 3, 40, int), ("F1o_dop", 1, 120, int), ("F1o_freq", 8, 15, float),
    ("Eo_vh", 8, 15, float), ("Eo_th", 8, 15, float), ("Eo_amp", 3, 40, int), ("Eo_dop", 1, 120, int), ("Eo_freq", 8, 15, float),
    ("F2x_vh", 8, 15, float), ("F2x_amp", 3, 40, int), ("F2x_dop", 1, 120, int), ("F2x_freq", 8, 15, float),
    ("F1x_vh", 8, 15, float), ("F1x_amp", 3, 40, int), ("F1x_dop", 1, 120, int), ("F1x_freq", 8, 15, float),
    ("Ex_vh", 8, 15, float), ("Ex_amp", 3, 40, int), ("Ex_dop", 1, 120, int), ("Ex_freq", 8, 15, float),
    ("median_amp_F", 3, 40, int), ("median_amp_E", 3, 40, int), ("median_amp_Es", 3, 40, int),
    ("th_coef_F2", 11, 10, float), ("th_coef_F1", 11, 10, float), ("th_coef_E", 11, 10, float),
    ("qp_segments", 20, 6, float), ("edit_flags", 1, 120, int), ("valley", 11, 10, float),
    ("Es_vh", 8, 15, float), ("Es_amp", 3, 40, int), ("Es_dop", 1, 120, int), ("Es_freq", 8, 15, float),
    ("Ea_vh", 8, 15, float), ("Ea_amp", 3, 40, int), ("Ea_dop", 1, 120, int), ("Ea_freq", 8, 15, float),
    ("profile_h", 8, 15, float), ("profile_fp", 8, 15, float), ("profile_ne", 8, 15, float),
    ("qual_letters", 1, 120, str), ("desc_letters", 1, 120, str), ("profile_edit_flags", 1, 120, int),
    ("th_coef_Ea", 11, 10, float), ("Ea_profile_h", 8, 15, float), ("Ea_profile_fp", 8, 15, float), ("Ea_profile_ne", 8, 15, float),
]

SAO_SCALED_NAMES = [  # 49 стандартных характеристик SAO 4.3 (группа 4)
    "foF2", "foF1", "M3000F2", "MUF3000F2", "fmin", "foEs", "fminF", "fminE", "foE", "fxI",
    "hF", "hF2", "hE", "hEs", "zmE", "yE", "QF", "QE", "DownF", "DownE", "DownEs", "FF", "FE", "D",
    "fMUF", "h_fMUF", "delta_foF2", "foEp", "f_hF", "f_hF2", "foF1p", "hmF2", "hmF1", "zhalfNm",
    "foF2p", "fminEs", "yF2", "yF1", "TEC", "scaleF2", "B0", "B1", "D1", "foEa", "hEa", "foP", "hP", "fbEs", "typeEs",
]


def read_sao(path: str | Path) -> dict:
    """Парсер SAO 4.x (однозаписный файл). Возвращает dict групп + 'scaled' как Series с именами."""
    lines = Path(path).read_text(errors="replace").splitlines()
    idx = [int(lines[0][i:i + 3]) for i in range(0, len(lines[0].rstrip("\n")), 3)]
    idx += [int(lines[1][i:i + 3]) for i in range(0, len(lines[1].rstrip("\n")), 3)]
    idx = idx[:80]
    out, ln = {}, 2
    for g, (name, width, per_line, kind) in enumerate(SAO_GROUP_FORMATS):
        n = idx[g] if g < len(idx) else 0
        if n == 0:
            continue
        if kind is str and width == 120:
            # Для текстовых групп индекс кодирует ЛИБО число строк (system_desc:
            # у PQ052 многострочный комментарий ARTIST даёт n=3), ЛИБО число
            # символов (time_settings: n=77). Эвристика: малое n — строки.
            n_lines = n if n < 40 else -(-n // 120)
            out[name] = "\n".join(lines[ln:ln + n_lines])
            ln += n_lines
            continue
        n_lines = -(-n // per_line)
        chunk = lines[ln:ln + n_lines]
        ln += n_lines
        vals = []
        for row in chunk:
            row = row.rstrip("\n")
            vals += [row[i:i + width] for i in range(0, len(row), width)]
        vals = vals[:n]
        if kind is str:
            out[name] = vals
        else:
            out[name] = np.array([kind(v) if v.strip() else np.nan for v in vals])
    if "scaled" in out:
        names = SAO_SCALED_NAMES[:len(out["scaled"])]
        s = pd.Series(out["scaled"], index=names, dtype=float)
        s[s >= 9999] = np.nan
        out["scaled"] = s
    # FF yyyy ddd mm dd HH MM SS ... (SAO 4.3, группа 3)
    m = re.search(r"^FF(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", out.get("time_settings", ""), re.M)
    if m:
        y, doy, _mo, _dd, hh, mm, ss = map(int, m.groups())
        out["datetime"] = dt.datetime(y, 1, 1) + dt.timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
    return out


# ----------------------------------------------------------------------------- EDP
def read_edp(path: str | Path) -> tuple[dict, pd.DataFrame]:
    """EDP (ARTIST/NHPC Electron Density Profile, текст). Возвращает (шапка, профиль)."""
    lines = Path(path).read_text(errors="replace").splitlines()
    head_keys = lines[0].split()
    head_vals = lines[1].split()
    head = {k: float(v) for k, v in zip(head_keys, head_vals[:len(head_keys)])}
    head["extra"] = " ".join(head_vals[len(head_keys):])
    rows = []
    for row in lines[3:]:
        p = row.split()
        if len(p) < 5:
            continue
        rows.append(dict(height_km=float(p[0]), freq_mhz=float(p[1]), f_conf=float(p[2]),
                         ne_m3=float(p[3]), ne_conf=float(p[4]),
                         artist_h=float(p[5]) if len(p) > 5 else np.nan,
                         flag=p[6] if len(p) > 6 else ""))
    df = pd.DataFrame(rows)
    df = df.replace(-99.0, np.nan).replace(-0.99e2, np.nan)
    return head, df


# ----------------------------------------------------------------------------- DFT
def read_dft(path: str | Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """DFT (таблицы 5C-48…5C-50). Возвращает (шапка из LSB первого блока, амплитуды, фазы).

    amp, phase: массивы формы (n_blocks, 16, 128); амплитуда в 3/8 дБ (LSB отброшен).
    """
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    n_blocks = len(raw) // BLOCK
    blocks = raw[:n_blocks * BLOCK].reshape(n_blocks, 16, 2, 128)
    amp_bytes, phase = blocks[:, :, 0, :], blocks[:, :, 1, :]
    amp = (amp_bytes >> 1).astype(float) * 3 / 8
    # Заголовок — по 1 биту в LSB амплитуд первого блока, младший бит нибла первым
    bits = (amp_bytes[0].reshape(-1) & 1)
    nib = lambda i: int("".join(str(b) for b in bits[4 * i:4 * i + 4][::-1]), 2)
    hdr = dict(
        record_type=nib(0), year=2000 + nib(1) * 10 + nib(2), doy=nib(3) * 100 + nib(4) * 10 + nib(5),
        hour=nib(6) * 10 + nib(7), minute=nib(8) * 10 + nib(9), second=nib(10) * 10 + nib(11),
        schedule=nib(12), program=nib(13), drift_flag=hex(nib(14) * 16 + nib(15)), journal=nib(16),
        first_height_10km=nib(17), height_res_code=nib(18), n_heights_code=nib(19),
        start_freq_khz=(nib(20) * 1e5 + nib(21) * 1e4 + nib(22) * 1e3 + nib(23) * 100 + nib(24) * 10 + nib(25)) / 10,
    )
    return hdr, amp, phase
