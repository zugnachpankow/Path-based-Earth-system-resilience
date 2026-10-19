import os, sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # run from the repo root so data/ resolves
from emission_floor import build_floor_map

"""build the branch-off scenarios from the NGFS current policy scenario"""

df = pd.read_csv("data/processed/NGFS_historic_merged.csv")

# filter to current policies, REMIND only
mask_current_policy = df["scenario"].str.contains("Current_Policies")
df_current_policy = df[mask_current_policy].copy()

mask_model = df_current_policy["scenario"].str.contains("REMIND")
df_current = df_current_policy[mask_model].copy()

start_interp = 2020.5
final_year = 2300.5
interp_years = np.arange(start_interp, final_year + 0.1, 1.0)
interp_cols = [str(y) for y in interp_years]

non_year_cols = [c for c in df_current.columns if not c.replace('.', '', 1).isdigit()]

all_year_cols = [str(y) for y in interp_years]
target_cols = non_year_cols + all_year_cols

# reindex to the annual grid, adding NaNs between the native 5-yearly values
df_interp = df_current.reindex(columns=target_cols).copy()

# interpolate across columns; limit_direction='both' holds the last native value
# (2100.5) constant out to 2300.5 rather than extrapolating a trend
df_interp[all_year_cols] = df_interp[all_year_cols].astype(float).interpolate(axis=1, limit_direction='both')

df_interp[all_year_cols] = df_interp[all_year_cols].ffill(axis=1).bfill(axis=1)

year_cols = all_year_cols
years = np.array([float(c) for c in year_cols])

def apply_absolute_reduction(
    row,
    start_year,
    reduction_rate,
    year_cols,
    baseline=0.0,
):
    # subtract a constant yearly amount (rate * start value), never below baseline
    # the floor is the species' natural background; a species already at or below it
    # (including net-negative emitters like CO2 AFOLU) is held constant from the
    # start year, since a proportional reduction would make a negative start rise
    out = row.copy()
    years_float = np.array([float(y) for y in year_cols])
    values = row[year_cols].astype(float).values

    start_idx_arr = np.where(years_float >= start_year)[0]
    if len(start_idx_arr) == 0:
        start_idx = len(values) - 1
    else:
        start_idx = start_idx_arr[0]

    start_value = values[start_idx]

    if start_value <= baseline:
        # already at/below the natural floor -> hold constant
        values[start_idx + 1:] = start_value
    else:
        abs_amount = start_value * reduction_rate
        for i in range(start_idx + 1, len(values)):
            values[i] = max(values[i - 1] - abs_amount, baseline)

    out[year_cols] = values
    return out


reduction_rates = np.arange(0.01, 0.105, 0.01)    # 0% .. 10% inclusive
starting_years = np.arange(2020.5, 2055.5, 5.0)    # 2020.5 .. 2050.5 inclusive

# natural background floor per species (fair baseline_emissions in each variable's
# CSV unit); reductions clamp here, not at 0
baseline_map = build_floor_map(df_interp)

all_results = []

for r in reduction_rates:
    for sy in starting_years:
        df_abs = df_interp.apply(
            lambda row: apply_absolute_reduction(
                row, sy, r, year_cols,
                baseline=baseline_map.get(row["variable"], 0.0)
            ),
            axis=1
        )
        df_abs = df_abs.copy()
        df_abs["base_scenario"] = df_abs["scenario"]
        rate_tag = f"{r:.2f}"
        df_abs["scenario"] += f"__start{sy}_r{rate_tag}"
        all_results.append(df_abs)

df_scenarios = pd.concat(all_results, ignore_index=True, sort=False)

# non-year cols then year cols
non_year_cols_after = [c for c in df_scenarios.columns if not c.replace('.', '', 1).isdigit()]
df_scenarios = df_scenarios[non_year_cols_after + year_cols]

# pre-2020 current policy history
mask_cp = df["scenario"].str.contains("Current_Policies")
df_cp_pre2020 = df.loc[mask_cp].copy()

pre2020_cols = [c for c in df.columns if c.replace('.', '', 1).isdigit() and float(c) <= 2019.5]

df_generated = df_scenarios.copy()

# pre-2020 values depend only on region/variable/unit, so attach them in one merge
pre2020_lookup = (
    df_cp_pre2020[['region', 'variable', 'unit'] + pre2020_cols]
    .drop_duplicates(['region', 'variable', 'unit'])
)
df_final = df_generated.merge(
    pre2020_lookup, on=['region', 'variable', 'unit'], how='left', validate='many_to_one'
)

df_final = df_final[['scenario', 'region', 'variable', 'unit'] + pre2020_cols + year_cols]

df_final.to_csv("data/processed/current_policy_branch_offs_2300.csv", index=False)

# corresponding volcanic and solar forcing files
df_vs = pd.read_csv("data/raw/volcanic_solar.csv")

all_scenarios = df_scenarios["scenario"].unique()

template_volcanic = df_vs[df_vs["Variable"] == "Volcanic"].iloc[0:1]
template_solar = df_vs[df_vs["Variable"] == "Solar"].iloc[0:1]

new_rows = []

for sc in all_scenarios:
    for var, template in [("Volcanic", template_volcanic), ("Solar", template_solar)]:
        if not ((df_vs["Scenario"] == sc) & (df_vs["Variable"] == var)).any():
            new_rows.append(template.assign(Scenario=sc))

df_vs_expanded = pd.concat([df_vs] + new_rows, ignore_index=True)
df_vs_expanded = df_vs_expanded.drop_duplicates()

df_vs_expanded.to_csv("data/processed/current_policy_branch_offs_2300_forcing.csv", index=False)