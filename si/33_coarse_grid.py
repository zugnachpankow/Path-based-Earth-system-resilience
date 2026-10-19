"""coarse screening grid of path-based resilience over three criterion thresholds, per ssp; in: all_scenarios_temperature.nc, tipping/*.nc; out: output/sensitivity/coarse_grid_{df,pivot,corr}.csv"""
import os
import sys
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from resilience import load_temperature, compute_running_means, load_tipping
from sensitivity import resilience_index

TEMPERATURE_FILE = "output/all_scenarios_temperature.nc"
TIPPING_DIR      = "output/tipping"
SENS_DIR         = "output/sensitivity"; os.makedirs(SENS_DIR, exist_ok=True)

SSPS = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"]

# grid
RES_X_VALS = [2080, 2090, 2100, 2110, 2120]
RES_Y_VALS = np.linspace(1.3, 2.0, 10)
RATE_VALS  = np.linspace(0.03, 0.06, 10)

# compute the grid one scenario at a time (keeps memory ~1 scenario)
temperature = load_temperature(TEMPERATURE_FILE)
tipping = load_tipping(TIPPING_DIR)

results = {}
for sc in tqdm(SSPS, desc="coarse grid"):
    rm = compute_running_means(temperature, [sc])[sc]
    tip_prob = tipping[sc]["prob_any_tipping"]   # no fillna(0): nan-aware mean in resilience_index
    grid = []
    for rx, ry, r in itertools.product(RES_X_VALS, RES_Y_VALS, RATE_VALS):
        val = resilience_index(rm, int(rx), float(ry), float(r), True, tip_prob)
        grid.append({"scenario": sc, "res_x": int(rx), "res_y": float(ry),
                     "rate": float(r), "resilience": val})
    results[sc] = pd.DataFrame(grid)

grid_df = pd.concat(results.values(), ignore_index=True)
pivot = grid_df.pivot_table(index=["res_x", "res_y", "rate"], columns="scenario", values="resilience")
corr = pivot[[c for c in SSPS if c in pivot.columns]].corr()

grid_df.to_csv(os.path.join(SENS_DIR, "coarse_grid_df.csv"), index=False)
pivot.to_csv(os.path.join(SENS_DIR, "coarse_grid_pivot.csv"))
corr.to_csv(os.path.join(SENS_DIR, "coarse_grid_corr.csv"))
print(f"\nSaved coarse grid -> {SENS_DIR}/coarse_grid_{{df,pivot,corr}}.csv", flush=True)
