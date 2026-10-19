"""minimum additional NDC reduction for a target resilience gain; in: 09 summary; out: finder_results_gain<G>.csv"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve data/ and output/ from the repo root

from calculator import (
    evaluate_candidate,
    gain_brackets_target,
    assert_configs_match_ensemble,
    N_FAIR_RUNS_BISECT,
    N_TIPPING_SAMPLES_BISECT,
    N_TIPPING_SAMPLES_FINAL,
)
from tipping_params import sample_lhs_params
from fair_config import FAIR_PARAMS, SPECIES_CONFIGS_NGFS

# configuration
# positional gain, e.g. `python 12_finder.py 0.10`
TARGET_GAIN     = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
TEMP_THRESHOLD  = 1.5
RATE_THRESHOLD  = 0.04
YEAR_TARGET     = 2100
TARGET_YEARS    = [2030.5, 2035.5, 2040.5]

BASE_SCENARIO       = "REMIND-MAgPIE_3.3-4.8___Nationally_Determined_Contributions_(NDCs)"
BASE_SCENARIO_CLEAN = BASE_SCENARIO.replace("(", "").replace(")", "")

EMISSIONS_FILE = "data/processed/NGFS_historic_merged_extended_to_2110.csv"
FORCING_FILE   = "data/processed/NGFS_forcing.csv"
PARAMS_FILE    = FAIR_PARAMS            # single calibration (v1.4.0)
SPECIES_FILE   = SPECIES_CONFIGS_NGFS

# initial guesses for R per gain, used to seed the bracket so the finder only refines
SEEDS_BY_GAIN = {
    0.01: "data/reference/finder_seeds_gain0.01.csv",
    0.10: "data/reference/finder_seeds_gain0.10.csv",
}
SEEDS_CSV = SEEDS_BY_GAIN.get(round(TARGET_GAIN, 2))
WORKSPACE = "output/calculator/finder"

# single-stage bisection; bracket centred on the seed but widened since the baseline
# differs from the one the seeds were tuned to
BRACKET  = 0.06     # ± around the seed reduction fraction
TOL      = 0.002
MAX_ITER = 10
# 10 stochastic realizations average internal variability; every run spans all 841
# configs (full ECS distribution), so n_runs only samples weather noise
N_FINDER_RUNS    = 10
N_FINDER_SAMPLES = N_TIPPING_SAMPLES_BISECT

# canonical NDC baseline resilience (from 09)
rate_str = f"{int(round(RATE_THRESHOLD * 100)):02d}"
_RBASE = "output/resilience"
summary_csv = (
    f"{_RBASE}/resilience_conditions_time{YEAR_TARGET}"
    f"_temp{TEMP_THRESHOLD}_rate{rate_str}/summary/"
    f"resilience_summary_time{YEAR_TARGET}_temp{TEMP_THRESHOLD}_rate{rate_str}.csv"
)
summary = pd.read_csv(summary_csv)
row = summary[summary["scenario"] == BASE_SCENARIO_CLEAN]
if row.empty:
    raise ValueError(f"NDC baseline not found in {summary_csv}")
baseline_resilience = float(row["resilience"].values[0])
target_resilience   = baseline_resilience + TARGET_GAIN
print(f"NDC baseline resilience = {baseline_resilience:.6f}  "
      f"-> target = {target_resilience:.6f} (+{TARGET_GAIN})")

# seeds: initial guesses for R
seeds = {}
if SEEDS_CSV and os.path.exists(SEEDS_CSV):
    sdf = pd.read_csv(SEEDS_CSV)
    seeds = dict(zip(sdf["target_year"].astype(float), sdf["converged_reduction_frac"].astype(float)))
    print(f"Loaded seeds for gain {TARGET_GAIN}: {seeds}")
else:
    print(f"[warn] seed file {SEEDS_CSV} not found — using 0.15 bracket centre.")

# emissions / forcing
df_emissions = pd.read_csv(EMISSIONS_FILE)
year_cols = [c for c in df_emissions.columns if c.replace(".", "", 1).isdigit()]
years = np.array([float(c) for c in year_cols])
df_forcing = pd.read_csv(FORCING_FILE)

# separate 100-point LHS for the finder, distinct from the main run's 1000-sample LHS
FINDER_LHS_SEED = 4321
assert_configs_match_ensemble()   # params file must match the ensemble's config set
lhs_params = sample_lhs_params(n_samples=N_FINDER_SAMPLES, seed=FINDER_LHS_SEED)

# bisection per target year; seed the bracket where available, else fall back to [0, 0.99]
results = []
for ty in TARGET_YEARS:
    if ty in seeds:
        seed = seeds[ty]
        lo, hi = max(0.0, seed - BRACKET), seed + BRACKET
    else:
        seed = float("nan")
        lo, hi = 0.0, 0.99
    print(f"\n══ target_year={ty}  seed={seed:.4f}  bracket=[{lo:.4f}, {hi:.4f}] ══")

    def _eval(R, label):
        iter_dir = os.path.join(WORKSPACE, f"gain{TARGET_GAIN}", f"ty{int(ty)}", label)
        return evaluate_candidate(
            R, df_emissions, years, year_cols,
            BASE_SCENARIO, ty, df_forcing,
            PARAMS_FILE, SPECIES_FILE, iter_dir,
            n_runs=N_FINDER_RUNS, n_samples=N_FINDER_SAMPLES,
            temp_threshold=TEMP_THRESHOLD, rate_threshold=RATE_THRESHOLD,
            year_target=YEAR_TARGET, lhs_params=lhs_params,
            candidate_name=f"NDC_calc__ty{int(ty)}_r{R:.6f}",
        )

    # monotone gain(R) must straddle TARGET_GAIN across the seeded bracket, else
    # bisection converges to an edge; widen to [0, 0.99] and error if that still fails
    g_lo = _eval(lo, "bracket_lo")["achieved_gain"]
    g_hi = _eval(hi, "bracket_hi")["achieved_gain"]
    if not gain_brackets_target(g_lo, g_hi, TARGET_GAIN):
        print(f"  [warn] bracket [{lo:.4f},{hi:.4f}] gains ({g_lo:.4f},{g_hi:.4f}) do not "
              f"straddle target {TARGET_GAIN}; widening to [0, 0.99]")
        lo, hi = 0.0, 0.99
        g_lo = _eval(lo, "bracket_lo_wide")["achieved_gain"]
        g_hi = _eval(hi, "bracket_hi_wide")["achieved_gain"]
        if not gain_brackets_target(g_lo, g_hi, TARGET_GAIN):
            raise RuntimeError(
                f"target gain {TARGET_GAIN} not bracketed on [0, 0.99] for ty={ty} "
                f"(gain(0)={g_lo:.4f}, gain(0.99)={g_hi:.4f})"
            )

    achieved_resilience = np.nan
    achieved_gain = np.nan
    history = []
    it = 0
    for it in range(MAX_ITER):
        mid = 0.5 * (lo + hi)
        iter_dir = os.path.join(WORKSPACE, f"gain{TARGET_GAIN}", f"ty{int(ty)}", f"iter_{it:02d}")
        res = evaluate_candidate(
            mid, df_emissions, years, year_cols,
            BASE_SCENARIO, ty, df_forcing,
            PARAMS_FILE, SPECIES_FILE, iter_dir,
            n_runs=N_FINDER_RUNS, n_samples=N_FINDER_SAMPLES,
            temp_threshold=TEMP_THRESHOLD, rate_threshold=RATE_THRESHOLD,
            year_target=YEAR_TARGET, lhs_params=lhs_params,
            candidate_name=f"NDC_calc__ty{int(ty)}_r{mid:.6f}",
        )
        achieved_resilience = res["achieved_resilience"]
        achieved_gain = res["achieved_gain"]   # paired candidate - NDC from the same run

        if achieved_gain < TARGET_GAIN:
            lo = mid
        else:
            hi = mid

        history.append({"iteration": it, "R": mid,
                        "achieved_resilience": achieved_resilience,
                        "achieved_gain": achieved_gain, "lo": lo, "hi": hi})
        print(f"  iter {it:02d}: R={mid:.6f}  resilience={achieved_resilience:.6f}  "
              f"gain={achieved_gain:.6f}  target={TARGET_GAIN}  [lo={lo:.4f}, hi={hi:.4f}]")

        if hi - lo < TOL:
            break

    converged = 0.5 * (lo + hi)
    # re-evaluate at the converged R so the reported resilience/gain/co2 belong to it
    res = _eval(converged, "converged")
    achieved_resilience = res["achieved_resilience"]
    achieved_gain = res["achieved_gain"]
    results.append({
        "target_year": ty,
        "converged_reduction_frac": converged,
        "achieved_resilience": achieved_resilience,
        "achieved_gain": achieved_gain,
        "ndc_resilience": res["ndc_resilience"],      # paired NDC (same run)
        "baseline_resilience": baseline_resilience,   # 09 summary value
        "co2_GtCO2": res["co2_GtCO2"],
        "reduction_vs_ndc2030_GtCO2": res["reduction_vs_ndc2030_GtCO2"],
        "reduction_vs_ndc2030_frac": res["reduction_vs_ndc2030_frac"],
        "ndc_2030_GtCO2": res["ndc_2030_GtCO2"],
        "target_gain": TARGET_GAIN,
        "temp_threshold": TEMP_THRESHOLD,
        "rate_threshold": RATE_THRESHOLD,
        "iterations": it + 1,
        "n_runs": N_FINDER_RUNS,
        "n_samples": N_FINDER_SAMPLES,
    })

    # incremental save after each target year
    os.makedirs(WORKSPACE, exist_ok=True)
    out_csv = os.path.join(WORKSPACE, f"finder_results_gain{TARGET_GAIN}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)

print(f"\nFinder complete. Results -> {out_csv}")
