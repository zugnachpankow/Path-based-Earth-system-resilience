"""combined mc/sensitivity table: block 1 coarse-grid per-ssp, block 2 inter-scenario correlation, block 3 lhs monte-carlo; in: coarse_grid_{df,corr}.csv, mc_*.pkl; out: tables/si_sensitivity.md + per-block csvs"""
import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import get_scenario_type

OUT_DIR  = os.path.join("tables"); os.makedirs(OUT_DIR, exist_ok=True)
SENS_DIR = "output/sensitivity"
MC_DIR   = os.path.join(SENS_DIR, "rate0.025_0.055")
FOCAL = ["ssp126", "ssp245", "ssp534-over"]
SSPS  = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"]
PERMISSIVE = np.array([2150.0, 2.0, 0.055])   # return year, temp, rate
STRICT     = np.array([2080.0, 1.3, 0.025])

md = ["# SI — Monte-Carlo / sensitivity of path-based resilience (Methods 6.4)\n"]

# blocks 1 & 2: coarse grid
CG_DF = os.path.join(SENS_DIR, "coarse_grid_df.csv")
CG_CORR = os.path.join(SENS_DIR, "coarse_grid_corr.csv")
if os.path.exists(CG_DF):
    grid_df = pd.read_csv(CG_DF)
    g = grid_df.groupby("scenario")["resilience"]
    block1 = pd.DataFrame({
        "min": g.min(), "median": g.median(), "max": g.max(),
        "share_gt0": grid_df.assign(pos=grid_df.resilience > 0).groupby("scenario")["pos"].mean(),
    }).reindex([s for s in SSPS if s in grid_df.scenario.unique()]).round(4)
    block1.to_csv(os.path.join(OUT_DIR, "si_sensitivity_block1_coarse.csv"))
    if os.path.exists(CG_CORR):
        corr = pd.read_csv(CG_CORR, index_col=0)
    else:
        corr = grid_df.pivot_table(index=["res_x", "res_y", "rate"], columns="scenario",
                                   values="resilience").corr()
    corr.round(3).to_csv(os.path.join(OUT_DIR, "si_sensitivity_block2_corr.csv"))
    md.append("## Block 1 — coarse screening grid (resilience per SSP)\n")
    md.append("Grid: return year [2080–2120]×5, temperature 1.3–2.0 °C ×10, rate 0.03–0.06 °C/yr ×10 "
              "(500 points/scenario), with tipping.\n")
    md.append(block1.reset_index().rename(columns={"index": "scenario"}).to_markdown(index=False) + "\n")
    md.append("## Block 2 — inter-scenario Pearson correlation across the grid\n")
    md.append("(NaN = constant-zero resilience over the whole grid, i.e. no variance.)\n")
    md.append(corr.round(3).to_markdown() + "\n")
    print("[Block 1/2] built from coarse grid:\n", block1.to_string(), flush=True)
else:
    md.append("## Block 1 & 2 — coarse screening grid\n")
    md.append("_Pending: run `33_coarse_grid.py` (SLURM: `run_33.sh`) to produce "
              "`output/sensitivity/coarse_grid_df.csv`._\n")
    print("[Block 1/2] coarse grid not found -> pending (run 33_coarse_grid.py).", flush=True)


# block 3: lhs monte-carlo
def _nearest(pm, corner):
    X = pm[:, :3].astype(float); rng = X.max(0) - X.min(0)
    return int(np.argmin(np.sqrt((((X - corner) / rng) ** 2).sum(1))))


def _cond_spearman_rate(pm, Y, n_bins=5):
    edges = np.percentile(pm[:, 1], np.linspace(0, 100, n_bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pm[:, 1] >= lo) & (pm[:, 1] < hi)
        rho = (np.nan if m.sum() < 30 else spearmanr(pm[m, 2], Y[m])[0])
        out.append((0.5 * (lo + hi), rho, int(m.sum())))
    return out


rows, cond_rows = [], []
for sc in FOCAL:
    with open(os.path.join(MC_DIR, f"mc_{sc}.pkl"), "rb") as f:
        d = pickle.load(f)
    pm = np.asarray(d["param_mc"]); Y = np.asarray(d["Y_mc"]); tip = pm[:, 3] > 0.5
    i_p, i_s = _nearest(pm, PERMISSIVE), _nearest(pm, STRICT)
    rows.append({
        "Scenario": get_scenario_type(sc),
        "default_tip_pct": round(float(d["baseline_tip"]) * 100, 3),
        "default_notip_pct": round(float(d["baseline_no_tip"]) * 100, 3),
        "mc_median_pct": round(float(np.median(Y)) * 100, 3),
        "mc_p5_pct": round(float(np.percentile(Y, 5)) * 100, 3),
        "mc_p95_pct": round(float(np.percentile(Y, 95)) * 100, 3),
        "permissive_corner_pct": round(float(Y[i_p]) * 100, 3),
        "strict_corner_pct": round(float(Y[i_s]) * 100, 3),
        "rho_return_year": round(float(spearmanr(pm[:, 0], Y)[0]), 3),
        "rho_temp": round(float(spearmanr(pm[:, 1], Y)[0]), 3),
        "rho_rate": round(float(spearmanr(pm[:, 2], Y)[0]), 3),
        "rho_tipping": round(float(spearmanr(pm[:, 3], Y)[0]), 3),
        "mean_with_tip_pct": round(float(Y[tip].mean()) * 100, 3),
        "mean_without_tip_pct": round(float(Y[~tip].mean()) * 100, 3),
    })
    for ctr, rho, n in _cond_spearman_rate(pm, Y):
        cond_rows.append({"Scenario": get_scenario_type(sc), "temp_quintile_center": round(ctr, 3),
                          "rho_rate_vs_resilience": (np.nan if np.isnan(rho) else round(rho, 3)), "n": n})

M = pd.DataFrame(rows); M.to_csv(os.path.join(OUT_DIR, "si_sensitivity_block3_mc.csv"), index=False)
C = pd.DataFrame(cond_rows); C.to_csv(os.path.join(OUT_DIR, "si_sensitivity_cond_spearman.csv"), index=False)

_labels = {
    "default_tip_pct": "Default, with tipping [%]", "default_notip_pct": "Default, no tipping [%]",
    "mc_median_pct": "MC median [%]", "mc_p5_pct": "MC 5th pct [%]", "mc_p95_pct": "MC 95th pct [%]",
    "permissive_corner_pct": "Permissive corner [%] †", "strict_corner_pct": "Strict corner [%] †",
    "rho_return_year": "ρ return year", "rho_temp": "ρ temperature", "rho_rate": "ρ rate",
    "rho_tipping": "ρ tipping", "mean_with_tip_pct": "Mean, with tipping [%]",
    "mean_without_tip_pct": "Mean, no tipping [%]",
}
disp = M.set_index("Scenario").T.rename(index=_labels).reset_index().rename(columns={"index": "Metric"})
piv = C.pivot_table(index="temp_quintile_center", columns="Scenario", values="rho_rate_vs_resilience").reset_index()

md.append("## Block 3 — LHS Monte-Carlo (N=10,000)\n")
md.append("Constraints sampled: return year 2080–2150, temperature 1.3–2.0 °C, "
          "warming rate 0.25–0.55 °C/decade, tipping on/off.\n")
md.append(disp.to_markdown(index=False) + "\n")
md.append("† nearest MC draw to the corner (the exact corner is not a stored evaluation).\n")
md.append("### Conditional Spearman ρ(rate, resilience) within temperature-threshold quintiles "
          "(Extended Data Fig. ed-sensitivity c)\n")
md.append(piv.round(3).to_markdown(index=False) + "\n")

with open(os.path.join(OUT_DIR, "si_sensitivity.md"), "w") as fh:
    fh.write("\n".join(md) + "\n")

print("\n[Block 3] MC table:\n", M.to_string(index=False), flush=True)
print("\nCombined table -> output/tables/si_sensitivity.md", flush=True)
