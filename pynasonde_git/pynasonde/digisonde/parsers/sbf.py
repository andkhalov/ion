"""Binary SBF format parser for Digisonde ionogram data.

This module provides :class:`SbfExtractor`, a parser for the SBF (Scaled
Binary Format) files produced by Digisonde DPS4D instruments. SBF shares its
fixed-block binary structure with RSF; the common byte-reading and tabular
flattening logic lives in the internal ``_rsf_sbf_base`` module.
"""

from pynasonde.digisonde.datatypes.sbfdatatypes import (
    SbfDataFile,
    SbfDataUnit,
    SbfFreuencyGroup,
    SbfHeader,
)
from pynasonde.digisonde.digi_utils import RSF_SBF_IONOGRAM_SETTINGS
from pynasonde.digisonde.parsers._rsf_sbf_base import RsfSbfBinaryBlockExtractor

SBF_IONOGRAM_SETTINGS = RSF_SBF_IONOGRAM_SETTINGS  # backwards-compatible alias


class SbfExtractor(RsfSbfBinaryBlockExtractor):
    """Low-level reader for SBF-format files."""

    data_file_class = SbfDataFile
    data_unit_class = SbfDataUnit
    header_class = SbfHeader
    frequency_group_class = SbfFreuencyGroup
    # Таблица 5C-44 мануала DPS-4D: SBF хранит 1 байт на бин дальности
    ionogram_settings = {
        "128": dict(number_freq_blocks=30, number_range_bins=128, byte_length=134),
        "256": dict(number_freq_blocks=15, number_range_bins=256, byte_length=262),
        "512": dict(number_freq_blocks=8, number_range_bins=498, byte_length=504),
    }
    bytes_per_bin = 1
    data_attr = "sbf_data"
    units_attr = "sbf_data_units"
    format_label = "SBF"
    include_azm_directions = False
    log_block_reads = True
    empty_returns_dataframe = False
