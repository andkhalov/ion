# -*- coding: utf-8 -*-
"""Манифест и потоковый лоадер (Э3 §2.1–2.2): опись по образцам, формы/типы тензоров, DataLoader."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from pyon import canon, loader, manifest
from pyon import oblique_synth as obs

from conftest import RSF_DIR


@pytest.fixture(scope="module")
def df_manifest():
    if not (RSF_DIR / "ionogram").exists():
        pytest.skip("нет образцов")
    df = manifest.build_manifest(procs=2, limit=40)      # первые 40 файлов = RSF-образцы JI91J
    return df


def test_manifest_columns_and_split(df_manifest):
    df = df_manifest
    assert len(df) == 40 and (df.station == "JI91J").all() and (df.fmt == "RSF").all()
    for col in ["path", "sao", "time", "foF2", "hF", "fxI", "M3000F2", "c_level", "model", "disturbed", "split"]:
        assert col in df.columns
    t = pd.to_datetime(df.time)
    assert t.is_monotonic_increasing and t.iloc[0] == pd.Timestamp("2022-01-01 00:00:00")
    assert set(df.split) == {"train", "val"} and (df.split == "val").sum() == 10   # 25 % хронологически
    assert (df.model == "DPS-4").all() and df.c_level.between(11, 55).all()


def test_geomag_join(df_manifest):
    """Индексы GFZ: Ap/Kp суток есть в манифесте; 2022-01-01 — спокойные сутки (Ap < 30)."""
    geo = manifest.load_geomag()
    if geo is None:
        pytest.skip("нет data/geomag/Kp_ap_Ap_SN_F107_since_1932.txt")
    assert {"Ap", "Kp_max", "F107", "disturbed_geo", "disturbed_letters", "disturbed", "daynight", "lt_hour"} <= set(df_manifest.columns)
    assert geo.loc["2022-01-01", "Ap"] == df_manifest.Ap.iloc[0] and 0 <= df_manifest.Kp_max.iloc[0] <= 9
    assert df_manifest.disturbed_geo.iloc[0] == int(geo.loc["2022-01-01", "Ap"] >= 30 or geo.loc["2022-01-01", "Kp_max"] >= 5)
    assert geo.loc["2003-10-29", "Ap"] > 100                            # Halloween storm — санити файла
    assert set(df_manifest.daynight) <= {"day", "night"}


def test_stem_time():
    st, t = manifest.stem_time("PQ052_2019365235959.SAO")
    assert st == "PQ052" and t == pd.Timestamp("2019-12-31 23:59:59")
    assert manifest.stem_time("garbage.RSF") is None


def test_vertical_dataset_item(df_manifest):
    ds = loader.VerticalDataset(df_manifest)
    x, y = ds[0]
    assert x.shape == (2, canon.NH, canon.NF) and x.dtype == torch.uint8 and int(x.max()) > 0
    assert y.shape == (canon.NH, canon.NF) and y.dtype == torch.int8
    assert set(torch.unique(y).tolist()) <= set(range(len(canon.CLASSES)))
    assert len(ds) == len(df_manifest)


def test_oblique_dataset_item(df_manifest):
    ds = loader.ObliqueDataset(df_manifest, component="O", fixed_d=800.0)
    y, d, m1, m2 = ds[0]
    assert y.shape == (obs.NP, obs.NF) and y.dtype == torch.int8 and float(d) == 800.0
    assert float(m1) > float(m2) > 0                                  # S1 на метках
    ds_rand = loader.ObliqueDataset(df_manifest, seed=0)
    d0 = float(ds_rand[3][1]); assert d0 in obs.D_SET
    assert float(loader.ObliqueDataset(df_manifest, seed=0)[3][1]) == d0   # воспроизводимо при том же seed


def test_dataloader_batches_with_workers(df_manifest):
    dl = DataLoader(loader.VerticalDataset(df_manifest), batch_size=8, num_workers=2)
    x, y = next(iter(dl))
    assert x.shape == (8, 2, canon.NH, canon.NF) and y.shape == (8, canon.NH, canon.NF)
    xf = x.float().div_(255)                                           # конвертация как в тренировочном цикле
    assert 0.0 <= float(xf.min()) and float(xf.max()) <= 1.0


def test_broken_file_yields_background(tmp_path, df_manifest):
    df = df_manifest.head(1).copy()
    bad = tmp_path / "bad.RSF"; bad.write_bytes(b"\x00" * 100)
    df.loc[df.index[0], "path"] = str(bad)
    x, y = loader.VerticalDataset(df)[0]
    assert int(x.max()) == 0 and int(y.max()) == 0
