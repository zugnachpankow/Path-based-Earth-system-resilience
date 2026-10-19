"""crossed-bootstrap 95% cis for the seven upset-bar criteria; reads running means (08) + tipping (07); writes output/bootstrap_upset. heavy, submit via SLURM

the seven criteria and their tensor shapes:
  C       no climate violation        (n_cfg, n_runs, 1)
  R       no rate violation           (n_cfg, n_runs, 1)
  T       no tipping                  (n_cfg, n_runs, n_samp)
  C+R     climate AND rate            (n_cfg, n_runs, 1)
  C+T     climate AND no tipping      (n_cfg, n_runs, n_samp)
  R+T     rate AND no tipping         (n_cfg, n_runs, n_samp)
  C+R+T   all three (= resilience)    (n_cfg, n_runs, n_samp)
criteria without a sample dimension (C, R, C+R) are broadcast to (n_cfg, n_runs, 1) so
the same bootstrap handles all seven.
"""
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve output/ from the repo root
from resilience import BOOTSTRAP_SEED   # shared bootstrap seed

RUNNING_MEAN_PATH = "output/running_mean_temps.pkl"
TIPPING_DIR       = "output/tipping"
OUT_DIR           = "output/bootstrap_upset"
os.makedirs(OUT_DIR, exist_ok=True)

# resilience conditions (must match 09 / figures)
res_x = 2100
res_y = 1.5
rate  = 0.04   # K/yr; 0.4 K/decade

SCENARIOS = [
    "REMIND-MAgPIE_3.3-4.8___Net_Zero_2050",
    "ssp119",
    "ssp126",
    "ssp534-over",
]
B = 1000


def crossed_bootstrap(T: np.ndarray, B: int = 1000, seed: int = 0) -> dict:
    """crossed bootstrap resampling (config, run, sample); returns mean and 95% ci"""
    rng = np.random.default_rng(seed)
    n_cfg, n_runs, n_samp = T.shape

    cfg_idx_all  = rng.integers(0, n_cfg,  size=(B, n_cfg))
    run_idx_all  = rng.integers(0, n_runs, size=(B, n_runs))
    samp_idx_all = rng.integers(0, n_samp, size=(B, n_samp))

    Q = np.empty(B, dtype=np.float64)
    for b in range(B):
        Q[b] = T[np.ix_(cfg_idx_all[b], run_idx_all[b], samp_idx_all[b])].mean()

    return {
        "mean":    float(Q.mean()),
        "ci_low":  float(np.quantile(Q, 0.025)),
        "ci_high": float(np.quantile(Q, 0.975)),
    }


# load once, shared across scenarios
print("loading running means and tipping data", flush=True)
with open(RUNNING_MEAN_PATH, "rb") as fh:
    running_mean_temps = pickle.load(fh)
running_mean_temps = {
    k.replace("(", "").replace(")", ""): v
    for k, v in running_mean_temps.items()
}

tipping_dict = {}
for fname in sorted(os.listdir(TIPPING_DIR)):
    if fname.endswith("_tipping_probabilities.nc"):
        sc = fname.split("_tipping_probabilities.nc")[0]
        tipping_dict[sc] = xr.open_dataset(os.path.join(TIPPING_DIR, fname))

all_rows = []

for SCENARIO in SCENARIOS:
    print(f"scenario: {SCENARIO}", flush=True)

    if SCENARIO not in running_mean_temps or SCENARIO not in tipping_dict:
        missing = [n for n, d in (("running mean temps", running_mean_temps),
                                  ("tipping", tipping_dict)) if SCENARIO not in d]
        print(f"  skip, missing: {missing}", flush=True)
        continue

    # resumable: reuse this scenario's csv if already written
    out_file = os.path.join(OUT_DIR, f"upset_bootstrap_{SCENARIO.replace('/', '_')}_B{B}.csv")
    if os.path.exists(out_file):
        all_rows.extend(pd.read_csv(out_file).to_dict("records"))
        print(f"  skip, already done ({out_file})", flush=True)
        continue

    da = tipping_dict[SCENARIO]["prob_any_tipping_sample"]   # (samp, cfg, run)

    rm = running_mean_temps[SCENARIO]

    climate_viol = (rm.sel(timebounds=slice(res_x, None)) > res_y).any(dim="timebounds")
    rate_viol    = (rm.diff(dim="timebounds")
                      .rolling(timebounds=10, center=True).mean()
                      .sel(timebounds=slice(2000, 2096)) > rate
                   ).any(dim="timebounds")

    no_climate_da    = (~climate_viol).astype(float).set_index(member=["config", "run"]).unstack("member")
    no_rate_da       = (~rate_viol).astype(float).set_index(member=["config", "run"]).unstack("member")
    resilience_cr_da = no_climate_da * no_rate_da

    cfg_coords = da.config.values
    run_coords = da.run.values
    n_cfg  = len(cfg_coords)
    n_runs = len(run_coords)
    n_samp = da.sizes["sample"]

    no_climate_2d    = no_climate_da.sel(config=cfg_coords, run=run_coords).values
    no_rate_2d       = no_rate_da.sel(config=cfg_coords, run=run_coords).values
    resilience_cr_2d = resilience_cr_da.sel(config=cfg_coords, run=run_coords).values

    # (n_cfg, n_runs, n_samp)
    no_tipping_3d = (1.0 - da.transpose("config", "run", "sample").values).astype(np.float32)

    tensors = {
        "C":     no_climate_2d[:, :, np.newaxis].astype(np.float32),
        "R":     no_rate_2d[:, :, np.newaxis].astype(np.float32),
        "T":     no_tipping_3d,
        "C+R":   resilience_cr_2d[:, :, np.newaxis].astype(np.float32),
        "C+T":   (no_climate_2d[:, :, np.newaxis] * no_tipping_3d).astype(np.float32),
        "R+T":   (no_rate_2d[:, :, np.newaxis] * no_tipping_3d).astype(np.float32),
        "C+R+T": (resilience_cr_2d[:, :, np.newaxis] * no_tipping_3d).astype(np.float32),
    }

    rows = []
    for name, T in tensors.items():
        point_est = float(T.mean())
        t0 = time.perf_counter()
        result = crossed_bootstrap(T, B=B, seed=BOOTSTRAP_SEED)
        elapsed = time.perf_counter() - t0

        row = {
            "scenario":  SCENARIO,
            "criterion": name,
            "point_est": round(point_est * 100, 2),
            "boot_mean": round(result["mean"] * 100, 2),
            "ci_low":    round(result["ci_low"] * 100, 2),
            "ci_high":   round(result["ci_high"] * 100, 2),
            "ci_width":  round((result["ci_high"] - result["ci_low"]) * 100, 2),
            "runtime_s": round(elapsed, 1),
            "n_cfg": n_cfg, "n_runs": n_runs, "n_samp": int(T.shape[2]), "B": B,
        }
        rows.append(row)
        all_rows.append(row)
        print(
            f"    {name:5s}  {point_est*100:5.1f}%  "
            f"[{result['ci_low']*100:.1f}, {result['ci_high']*100:.1f}]  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    out_file = os.path.join(OUT_DIR, f"upset_bootstrap_{SCENARIO.replace('/', '_')}_B{B}.csv")
    pd.DataFrame(rows).to_csv(out_file, index=False)

combined_file = os.path.join(OUT_DIR, f"upset_bootstrap_combined_B{B}.csv")
df_all = pd.DataFrame(all_rows)
df_all.to_csv(combined_file, index=False)
print(df_all.to_string(index=False))
