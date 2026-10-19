"""clean the raw NGFS IAM workbook; in: data/NGFS_raw/IAM_data.xlsx; out: data/raw/NGFS_cleaned_IAM_all.csv"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

RAW_XLSX = "data/NGFS_raw/IAM_data.xlsx"
OUT_CSV  = "data/raw/NGFS_cleaned_IAM_all.csv"
SHEET    = "data"
# three NGFS IAMs, dropping the IntegratedPhysicalDamages variant
MODELS = [
    "GCAM 6.0 NGFS",
    "MESSAGEix-GLOBIOM 2.0-M-R12-NGFS",
    "REMIND-MAgPIE 3.3-4.8",
]
# infilled is the complete emulator-ready species set
PREFIX = "AR6 climate diagnostics|Infilled|Emissions|"


def _short(variable):
    # last '|' segment, except CO2 fossil/industry which becomes 'FFI'
    if variable.endswith("CO2|Energy and Industrial Processes"):
        return "FFI"
    return variable.split("|")[-1].strip()


def main():
    if not os.path.exists(RAW_XLSX):
        sys.exit(f"Raw NGFS workbook not found at {RAW_XLSX}. Download the NGFS IAM outputs "
                 f"(https://data.ece.iiasa.ac.at/ngfs) and place it there (sheet '{SHEET}').")
    raw = pd.read_excel(RAW_XLSX, sheet_name=SHEET)

    df = raw[
        (raw["Region"] == "World")
        & (raw["Model"].isin(MODELS))
        & (raw["Variable"].astype(str).str.startswith(PREFIX))
    ].copy()
    df["Variable"] = df["Variable"].map(_short)
    # strip the degree sign in some scenario names
    df["Scenario"] = df["Scenario"].astype(str).str.replace("°", "", regex=False)

    year_cols = [c for c in df.columns if str(c).isdigit()]
    df = df[["Model", "Scenario", "Region", "Variable", "Unit"] + year_cols]
    df = df.sort_values(["Model", "Scenario", "Variable"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}: {df.shape[0]} rows", flush=True)


if __name__ == "__main__":
    main()
