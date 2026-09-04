# -*- coding: utf-8 -*-
"""Общие фикстуры: пути к образцам Щирого (data/*samples*), словарь онтологии."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RSF_DIR = ROOT / "data/RSF-samples-w-img-n-sao-n-dft"
SBF_DIR = ROOT / "data/SBF-samples-w-img-n-sao"
JI = "JI91J_2022001000000"
RO = "RO041_2022001000000"


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"нет образца {p.relative_to(ROOT)} (перенести data/*samples*)")
    return p


@pytest.fixture(scope="session")
def rsf_path():
    return _need(RSF_DIR / f"ionogram/{JI}.RSF")


@pytest.fixture(scope="session")
def sbf_path():
    return _need(SBF_DIR / f"ionogram/{RO}.SBF")


@pytest.fixture(scope="session")
def sao_ji_path():
    return _need(RSF_DIR / f"scaled/{JI}.SAO")


@pytest.fixture(scope="session")
def sao_ro_path():
    return _need(SBF_DIR / f"scaled/{RO}.SAO")


@pytest.fixture(scope="session")
def edp_path():
    return _need(RSF_DIR / f"scaled/{JI}.EDP")


@pytest.fixture(scope="session")
def dft_path():
    files = sorted((RSF_DIR / "drift").glob("*.DFT"))
    if not files:
        pytest.skip("нет DFT-образцов")
    return files[0]


@pytest.fixture(scope="session")
def sao_ji(sao_ji_path):
    from pyon import digi_formats as dfm
    return dfm.read_sao(sao_ji_path)


@pytest.fixture(scope="session")
def sao_ro(sao_ro_path):
    from pyon import digi_formats as dfm
    return dfm.read_sao(sao_ro_path)


@pytest.fixture(scope="session")
def vocab():
    from pyon import validate as vd
    return vd.load_vocabulary()
