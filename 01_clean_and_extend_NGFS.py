import pandas as pd
import re
import numpy as np
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # run from the repo root so data/ resolves
from rcmip import clean_rcmip_historical_for_fair
from emission_floor import build_floor_map
from units import convert_emissions

"""clean and extend the NGFS scenarios for fair"""

os.makedirs("data/processed", exist_ok=True)

df_to_clean = pd.read_csv("data/raw/NGFS_cleaned_IAM_all.csv")

# lowercase headers; turn year ints into midyear (2000 -> 2000.5)
def transform_header(col):
    col_lower = col.lower()
    if re.fullmatch(r"\d{3,4}", col_lower):
        return f"{col_lower}.5"
    return col_lower


df_to_clean.columns = [transform_header(c) for c in df_to_clean.columns]

# replace whitespaces
df_to_clean[["model", "scenario"]] = (
    df_to_clean[["model", "scenario"]]
    .replace(r"\s+", "_", regex=True)
)

# merge model and scenario
df_to_clean["scenario"] = df_to_clean["model"] + "___" + df_to_clean["scenario"]
df_to_clean = df_to_clean.drop(columns=["model"])

fair_variables = pd.read_csv("data/raw/fair_variables_reference.csv")

fair_vars = set(fair_variables["variable"].unique())
ngfs_vars = set(df_to_clean["variable"].unique())

# add the dash in CFC/HCFC/HFC/Halon names to match fair
def rename_variable(var):
    v = var.strip()

    for prefix in ["CFC", "HCFC", "HFC", "Halon"]:
        if v.startswith(prefix) and not v.startswith(prefix + "-"):
            return re.sub(rf"^{prefix}(.*)", rf"{prefix}-\1", v)

    return v

df_to_clean["variable"] = df_to_clean["variable"].astype(str).apply(rename_variable)

rename_map = {"FFI": "CO2 FFI", "AFOLU": "CO2 AFOLU", "cC4F8": "c-C4F8", "HFC-43-10": "HFC-4310mee"}
df_to_clean["variable"] = df_to_clean["variable"].replace(rename_map)

# drop species not in fair
df_to_clean = df_to_clean[df_to_clean["variable"] != "Halon-1202"]
df_to_clean = df_to_clean[df_to_clean["variable"] != "HFC-245ca"]

fair_vars = set(fair_variables["variable"].unique())
ngfs_vars = set(df_to_clean["variable"].unique())

fair_units = pd.read_csv("data/raw/fair_units_reference.csv")

fair_un = set(fair_units["unit"].unique())

# CO2 and N2O units are converted later
def rename_unit(var):
    v = var.strip()

    if v == "kt HFC43-10/yr":
        return "kt HFC4310mee/yr"

    return v

df_to_clean["unit"] = df_to_clean["unit"].astype(str).apply(rename_unit)

# history from RCMIP ends at 2014.5; NGFS futures start at 2020.5. the 2015-2019 gap
# is left empty and filled by a linear bridge at fair ingestion (interp1d linear from
# the 2014.5 RCMIP value to the 2020.5 NGFS value)
df1 = clean_rcmip_historical_for_fair(species=fair_variables["variable"].tolist())
df2 = df_to_clean

fair_not_in_ngfs = fair_vars - ngfs_vars

# drop HFC-245fa, not in ngfs
df1 = df1[df1["variable"] != "HFC-245fa"]

scenarios = df2["scenario"].unique().tolist()

year_cols_df1 = [c for c in df1.columns if c.replace('.', '', 1).isdigit()]
year_cols_df2 = [c for c in df2.columns if c.replace('.', '', 1).isdigit()]

# replicate df1 for each scenario
df1_expanded = pd.concat(
    [df1.assign(scenario=sc) for sc in scenarios],
    ignore_index=True
)

# convert each NGFS variable to its fair target unit via fair's own unit tables;
# an unknown unit pair raises (NOx relabel Mt NO2/yr == Mt NOx/yr is handled natively)
unit_map = df1.groupby("variable")["unit"].first().to_dict()

for var, target_unit in unit_map.items():
    mask = df2["variable"] == var
    if mask.any():
        current_unit = df2.loc[mask, "unit"].iloc[0]
        if current_unit != target_unit:
            df2.loc[mask, year_cols_df2] = convert_emissions(
                df2.loc[mask, year_cols_df2].values, current_unit, target_unit
            )
            df2.loc[mask, "unit"] = target_unit

df2.to_csv("data/processed/NGFS_only_cleaned.csv", index=False)

ext_units = set(df1_expanded["unit"].unique())
ext_not_in_fair = ext_units - fair_un

# drop years > 2014.5 (RCMIP history ends here; NGFS futures resume at 2020.5)
df1_trimmed = df1_expanded.loc[:, ["scenario", "region", "variable", "unit"] +
                               [c for c in year_cols_df1 if float(c) <= 2014.5]]

# df2 timeline from 2020.5 onward
df2_years = [c for c in year_cols_df2 if float(c) >= 2020.5]

merge_keys = ["scenario", "region", "variable", "unit"]

df2_trimmed = df2[merge_keys + df2_years]

df_final = pd.merge(
    df1_trimmed,
    df2_trimmed,
    how="left",
    on=merge_keys,
    validate="one_to_one"
)

assert not df_final.duplicated(merge_keys).any()
check = df_final[df_final[df2_years].isna().any(axis=1)]
assert check.empty, check[merge_keys].drop_duplicates()

df_final = df_final.reset_index(drop=True)
df_final.to_csv("data/processed/NGFS_historic_merged.csv", index=False)

# rebuild volcanic and solar forcing files for the NGFS scenarios
df_vs = pd.read_csv("data/raw/volcanic_solar.csv")

template_volcanic = df_vs[df_vs["Variable"] == "Volcanic"].iloc[0:1]
template_solar = df_vs[df_vs["Variable"] == "Solar"].iloc[0:1]

new_rows = []

for sc in scenarios:
    for var, template in [("Volcanic", template_volcanic), ("Solar", template_solar)]:
        if not ((df_vs["Scenario"] == sc) & (df_vs["Variable"] == var)).any():
            new_rows.append(template.assign(Scenario=sc))

df_vs_expanded = pd.concat([df_vs] + new_rows, ignore_index=True)
df_vs_expanded = df_vs_expanded.drop_duplicates()

df_vs_expanded.to_csv("data/processed/NGFS_forcing.csv", index=False)

# extend 2 steps by extrapolation so the centred 20-yr running mean can reach 2100
def expand_for_running_mean(
    df,
    years,
    year_cols,
    interpolation_range_start=2090.5,
    interpolation_range_end=2100.5,
    extensions_length=2,
    floor_map=None,
):
    idx_start = np.argmin(np.abs(years - interpolation_range_start))
    idx_end   = np.argmin(np.abs(years - interpolation_range_end))

    last_year = float(year_cols[idx_end])
    extra_years = [last_year + 5*(i + 1) for i in range(extensions_length)]
    extra_cols = [f"{y:.1f}" for y in extra_years]

    new_rows = []

    for _, row in df.iterrows():
        base_vals = row[year_cols].astype(float).values

        # fit a line over the final decade and extrapolate
        x = years[idx_start:idx_end+1]
        y = base_vals[idx_start:idx_end+1]

        a, b = np.polyfit(x, y, 1)

        x_extra = np.array(extra_years)
        y_extra = a * x_extra + b

        # clamp the tail at the species' natural floor so a declining trend cannot
        # run below background; never clamp above the last real value, so already
        # sub-floor trajectories like net-negative CO2 AFOLU hold
        if floor_map is not None:
            floor_eff = min(floor_map.get(row["variable"], 0.0), base_vals[idx_end])
            y_extra = np.maximum(y_extra, floor_eff)

        new_row = row.copy()
        for col, val in zip(extra_cols, y_extra):
            new_row[col] = val

        new_rows.append(new_row)

    return pd.DataFrame(new_rows)

year_cols = [c for c in df_final.columns if c.replace('.', '', 1).isdigit()]
years = np.array([float(c) for c in year_cols])

df_extended = expand_for_running_mean(
    df_final,
    years,
    year_cols,
    interpolation_range_start=2090.5,
    interpolation_range_end=2100.5,
    extensions_length=2,
    floor_map=build_floor_map(df_final),
)

df_extended.to_csv("data/processed/NGFS_historic_merged_extended_to_2110.csv", index=False)