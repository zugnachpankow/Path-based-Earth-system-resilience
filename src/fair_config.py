"""shared FAIR calibration config: file paths, parameter loading and natural (solar + volcanic) forcing."""
import os
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd

_DATA_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "raw")

FAIR_PARAMS = os.path.join(_DATA_RAW, "calibrated_constrained_parameters_calibration1.4.0.csv")
SPECIES_CONFIGS = os.path.join(_DATA_RAW, "species_configs_properties_calibration1.4.0.csv")
SPECIES_CONFIGS_NGFS = os.path.join(_DATA_RAW, "species_configs_properties_NGFS_calibration1.4.0.csv")

SOLAR_ERF = os.path.join(_DATA_RAW, "solar_erf_timebounds.csv")
VOLCANIC_ERF = os.path.join(_DATA_RAW, "volcanic_ERF_1750-2101_timebounds.csv")

# non-fair columns carried in the parameter file, consumed here not by override_defaults
SOLAR_SCALE_COLS = ["fscale_solar_amplitude", "fscale_solar_trend"]

# solar trend ramps linearly 0->1 over 1750..2020, then stays at 1
_TREND_START, _TREND_END = 1750, 2020


def load_fair_params(path=FAIR_PARAMS):
    """read a calibration parameter file, splitting off the solar-scale columns into solar"""
    df = pd.read_csv(path, index_col=0)
    solar = df[SOLAR_SCALE_COLS].copy()
    df_fair = df.drop(columns=SOLAR_SCALE_COLS)
    return df_fair, solar


@contextmanager
def fair_override_file(df_fair):
    """write df_fair to a temporary csv for FAIR.override_defaults"""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    tmp.close()
    try:
        df_fair.to_csv(tmp.name)
        yield tmp.name
    finally:
        os.unlink(tmp.name)


def natural_forcing(timebounds, solar_file=SOLAR_ERF, volcanic_file=VOLCANIC_ERF):
    """solar and volcanic erf aligned to timebounds; volcanic zero outside its file range"""
    years = np.asarray(timebounds).astype(int)
    solar = pd.read_csv(solar_file, index_col="year")["erf"]
    volc = pd.read_csv(volcanic_file, index_col="timebounds")["erf"]

    solar_out = solar.reindex(years).values
    if np.isnan(solar_out).any():
        missing = years[np.isnan(solar_out)]
        raise ValueError(f"solar ERF undefined for years {missing[:5]}... "
                         f"(file covers {int(solar.index.min())}..{int(solar.index.max())})")

    volc_out = np.zeros(len(years))
    in_range = (years >= int(volc.index.min())) & (years <= int(volc.index.max()))
    volc_out[in_range] = volc.reindex(years[in_range]).values
    return volc_out, solar_out


def solar_trend_shape(timebounds):
    """linear 0->1 ramp over 1750..2020, flat at 1 afterwards"""
    years = np.asarray(timebounds).astype(float)
    shape = np.ones(len(years))
    ramp = (years >= _TREND_START) & (years <= _TREND_END)
    shape[ramp] = (years[ramp] - _TREND_START) / (_TREND_END - _TREND_START)
    shape[years < _TREND_START] = 0.0
    return shape
