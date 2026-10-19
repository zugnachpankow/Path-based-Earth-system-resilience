"""shared temperature series; in: raw GMST DataArray; out: per-member rebased 20-yr running mean."""
import numpy as np
import xarray as xr


def _baseline_weights(n):
    # half weights on the two endpoints for the 1850-1901 baseline
    w = np.ones(n); w[0] = w[-1] = 0.5
    return w


def rebase_per_member(da, baseline=(1850, 1901)):
    # subtract each member's weighted 1850-1900 mean (half weights on endpoints)
    base_slice = da.sel(timebounds=slice(*baseline))
    w = xr.DataArray(_baseline_weights(base_slice.timebounds.size),
                     dims="timebounds", coords={"timebounds": base_slice.timebounds})
    base = base_slice.weighted(w).mean("timebounds")
    return da - base


def running_mean(da, window=20):
    # 20-yr centred running mean; nan edges from the centred window dropped
    return da.rolling(timebounds=window, center=True).mean().dropna("timebounds", how="all")


def processed_temperature(da, window=20, baseline=(1850, 1901)):
    return running_mean(rebase_per_member(da, baseline), window)


def last_valid_year(da):
    other = [d for d in da.dims if d != "timebounds"]
    valid = da.notnull().any(other) if other else da.notnull()
    idx = np.where(valid.values)[0]
    return float(da.timebounds.values[idx[-1]]) if idx.size else float("nan")
