"""resilience post-processing; in: temperature/tipping files; out: resilience dataframes."""
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import xarray as xr

from temperature import processed_temperature


def load_temperature(path):
    da = xr.open_dataset(path)["__xarray_dataarray_variable__"]
    return da.assign_coords(
        scenario=da.scenario.str.replace(r"[\(\)]", "", regex=True)
    )


def load_tipping(tipping_dir):
    files = [
        f for f in listdir(tipping_dir)
        if f.endswith(".nc") and isfile(join(tipping_dir, f))
    ]
    out = {}
    for f in files:
        scenario = f.split("_tipping_probabilities.nc")[0]
        out[scenario] = xr.open_dataset(join(tipping_dir, f))
    return out


def compute_running_means(temperature, scenarios, window=20, baseline=(1850, 1901)):
    # per-member 1850-1900 rebase then 20-yr centred running mean; members are (run, config)
    running_means = {}
    for scenario in scenarios:
        data = temperature.sel(scenario=scenario).stack(member=("run", "config"))
        running_means[scenario] = processed_temperature(data, window=window, baseline=baseline)
    return running_means


def compute_resilience(rm, res_x, res_y, rate, res_end=None):
    # binary resilience: not resilient if GMST exceeds res_y in [res_x, res_end]
    # or 10-yr warming rate exceeds rate over 2000..2096
    climate_violation = (
        rm.sel(timebounds=slice(res_x, res_end)) > res_y
    ).any(dim="timebounds")
    rate_of_change = rm.diff("timebounds").rolling(timebounds=10, center=True).mean()
    rate_violation = (
        rate_of_change.sel(timebounds=slice(2000, 2096)) > rate
    ).any(dim="timebounds")

    binary = (~(climate_violation | rate_violation)).unstack("member").astype(float)
    df = binary.to_dataframe(name="resilience_cr").reset_index()
    return df.drop(columns=["layer"], errors="ignore")


BOOTSTRAP_SEED = 0


def crossed_bootstrap(T, B=1000, seed=BOOTSTRAP_SEED):
    # crossed bootstrap: resample config, run, sample independently then cross
    rng = np.random.default_rng(seed)
    n_cfg, n_runs, n_samp = T.shape
    Q = np.empty(B, dtype=np.float32)
    for b in range(B):
        cfg_idx = rng.integers(0, n_cfg, size=n_cfg)
        run_idx = rng.integers(0, n_runs, size=n_runs)
        samp_idx = rng.integers(0, n_samp, size=n_samp)
        # nan-aware: failed tipping solves are nan, dropped here
        Q[b] = np.nanmean(T[cfg_idx][:, run_idx][:, :, samp_idx])
    return {
        "mean": float(Q.mean()),
        "ci_low": float(np.quantile(Q, 0.025)),
        "ci_high": float(np.quantile(Q, 0.975)),
    }
