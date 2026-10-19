"""main resilience table per scenario, standard criterion; reads resilience summary; writes tables/main_resilience_table.{csv,md}."""
import os
import sys

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from figures_common import (
    RESILIENCE_BASE, FAIR_PARAMS_FILE, res_x, res_y, rate_str, build_naming, get_model,
)

OUT_DIR = os.path.join(_HERE, "tables")
os.makedirs(OUT_DIR, exist_ok=True)

# clopper-pearson N = number of configs (the independent unit)
N_CP = len(pd.read_csv(FAIR_PARAMS_FILE, index_col=0))

SUMMARY_CSV = os.path.join(
    RESILIENCE_BASE, f"resilience_conditions_time{res_x}_temp{res_y}_rate{rate_str}",
    "summary", f"resilience_summary_time{res_x}_temp{res_y}_rate{rate_str}.csv",
)

df = pd.read_csv(SUMMARY_CSV)
# keep real scenarios (ssps + ngfs); drop the current-policies branch-off grid
df = df[~df["scenario"].str.contains("GCAM|MESSAGE|Current_Policies__start", regex=True)].copy()
fancy_titles, _, _, _ = build_naming(df["scenario"].tolist())

def _family(sc):
    if sc.startswith("ssp"):
        return "SSP"
    if "Current_Policies__start" in sc:
        return "Branch-off"
    return "NGFS"

def _cp(p):
    lo, hi = proportion_confint(count=round(p * N_CP), nobs=N_CP, alpha=0.05, method="beta")
    return lo, hi

df["name"]   = df["scenario"].map(fancy_titles)
df["family"] = df["scenario"].map(_family)
# bootstrap and clopper-pearson cis, both in %
df["boot_lo_pct"] = df["resilience_ci_low"] * 100.0
df["boot_hi_pct"] = df["resilience_ci_high"] * 100.0
df["cp_lo_pct"]   = df["resilience"].map(lambda p: _cp(p)[0] * 100.0)
df["cp_hi_pct"]   = df["resilience"].map(lambda p: _cp(p)[1] * 100.0)
df["resilience_pct"] = df["resilience"] * 100.0
df = df.sort_values("resilience", ascending=False).reset_index(drop=True)

# numeric csv
cols = ["scenario", "name", "family", "resilience_pct",
        "boot_lo_pct", "boot_hi_pct", "cp_lo_pct", "cp_hi_pct"]
csv_path = os.path.join(OUT_DIR, "main_resilience_table.csv")
df[cols].to_csv(csv_path, index=False)
print(f"Wrote {csv_path}  ({len(df)} scenarios)  [N_CP={N_CP}]", flush=True)

# formatted markdown
md = df.copy()
md["Path-based resilience [%]"] = md["resilience_pct"].map(lambda v: f"{v:.1f}")
md["95% CI (bootstrap)"] = md.apply(lambda r: f"[{r['boot_lo_pct']:.1f}–{r['boot_hi_pct']:.1f}]", axis=1)
def _cp_str(r):
    # two decimals where the bootstrap collapses at zero
    nd = 2 if (r["boot_lo_pct"] == 0.0 and r["boot_hi_pct"] == 0.0) else 1
    return f"[{r['cp_lo_pct']:.{nd}f}–{r['cp_hi_pct']:.{nd}f}]"
md["95% CI (Clopper–Pearson)"] = md.apply(_cp_str, axis=1)
md = md[["name", "family", "Path-based resilience [%]",
         "95% CI (bootstrap)", "95% CI (Clopper–Pearson)"]].rename(
    columns={"name": "Scenario", "family": "Family"})
md_path = os.path.join(OUT_DIR, "main_resilience_table.md")
md.to_markdown(md_path, index=False)
print(f"Wrote {md_path}", flush=True)
print(f"\nCriterion: {res_y} °C by {res_x}, rate 0.{rate_str} °C/yr, incl. tipping "
      f"| Clopper–Pearson N={N_CP}")
print(md.to_string(index=False))
