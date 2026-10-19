"""emission-scenario construction shared by the calculator and the figures; emissions in, candidate scenario out."""
import numpy as np
import pandas as pd


def build_candidate_scenario(df_emissions, years, year_cols,
                             base_scenario_name, target_year,
                             reduction_frac, candidate_name,
                             ramp_start_year=2025.5, baseline_map=None):
    """apply an additional fractional reduction to the base scenario at target_year, carrying the offset forward

    baseline_map clamps each species at its natural floor (fair baseline_emissions);
    None clamps at 0. at reduction_frac == 0 the candidate reproduces the base exactly.
    """
    i_target = np.argmin(np.abs(years - target_year))

    df_base = df_emissions[df_emissions["scenario"] == base_scenario_name].copy()
    if df_base.empty:
        raise ValueError(f"Scenario '{base_scenario_name}' not found in emissions.")

    new_rows = []
    for _, row in df_base.iterrows():
        new_row  = row.copy()
        vals     = new_row[year_cols].astype(float).values.copy()
        v_target_old = vals[i_target]

        # ramped absolute offset subtracted from the base, preserving its shape;
        # R == 0 gives offset == 0 everywhere so candidate == base
        difference = v_target_old * reduction_frac          # absolute cut reached at target
        offset = np.zeros_like(vals)
        ramp_mask = (years >= ramp_start_year) & (years <= target_year)
        offset[ramp_mask] = np.linspace(0.0, difference, ramp_mask.sum())
        offset[years > target_year] = difference
        reduced = vals - offset

        # emission floor: never cut below natural background, and never raise the base
        flr = 0.0 if baseline_map is None else float(baseline_map.get(str(row["variable"]).strip(), 0.0))
        vals = np.minimum(vals, np.maximum(reduced, flr))

        new_row[year_cols] = vals
        new_row["scenario"] = candidate_name
        new_rows.append(new_row)

    return pd.DataFrame(new_rows)[df_emissions.columns].reset_index(drop=True)
