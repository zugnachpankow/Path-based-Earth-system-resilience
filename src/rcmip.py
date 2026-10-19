"""local rcmip readers for fair; in: rcmip csvs; out: emissions/concentration/forcing fills."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from fair.interface import fill
from fair.exceptions import MissingDataError
from fair.structure.units import (
    compound_convert,
    desired_concentration_units,
    desired_emissions_units,
    mixing_ratio_convert,
    prefix_convert,
    time_convert,
)

# rcmip source files in data/raw, downloaded manually (see README data availability)
_DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RCMIP_EMISSIONS = str(_DATA_RAW / "rcmip-emissions-annual-means-v5-1-0.csv")
RCMIP_CONCENTRATIONS = str(_DATA_RAW / "rcmip-concentrations-annual-means-v5-1-0.csv")
RCMIP_FORCING = str(_DATA_RAW / "rcmip-radiative-forcing-annual-means-v5-1-0.csv")

def build_species_map(species):
    """map fair species names to their rcmip variable-name suffixes; specials pruned to species."""
    m = {specie: specie.replace("-", "") for specie in species}
    m["CO2 FFI"] = "CO2|MAGICC Fossil and Industrial"
    m["CO2 AFOLU"] = "CO2|MAGICC AFOLU"
    m["NOx aviation"] = "NOx|MAGICC Fossil and Industrial|Aircraft"
    m["Aerosol-radiation interactions"] = "Aerosols-radiation interactions"
    m["Aerosol-cloud interactions"] = "Aerosols-radiation interactions"
    m["Contrails"] = "Contrails and Contrail-induced Cirrus"
    m["Light absorbing particles on snow and ice"] = "BC on Snow"
    m["Stratospheric water vapour"] = "CH4 Oxidation Stratospheric H2O"
    m["Land use"] = "Albedo Change"
    return {specie: name for specie, name in m.items() if specie in species}


# rcmip reports GFED biomass-burning NOx in Mt NO but labels it "Mt NOx/yr"; fair
# wants Mt NO2/yr, so scale the four GFED sectors by NO2/NO and add the already-NO2
# agriculture and fossil & industrial sectors (as fair-calibrate v1.4.0 does)
_NOX_GFED_SECTORS = [
    "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning",
    "Emissions|NOx|MAGICC AFOLU|Forest Burning",
    "Emissions|NOx|MAGICC AFOLU|Grassland Burning",
    "Emissions|NOx|MAGICC AFOLU|Peat Burning",
]
_NOX_AGRICULTURE = "Emissions|NOx|MAGICC AFOLU|Agriculture"
_NOX_FOSSIL_INDUSTRIAL = "Emissions|NOx|MAGICC Fossil and Industrial"
_NOX_NO2_PER_NO = 46.006 / 30.006


def reconstruct_nox_emissions(df_emis, scenario, region, year_cols):
    """total nox (Mt NO2/yr) over year_cols, gfed sectors NO->NO2 corrected."""
    m = (df_emis["Scenario"] == scenario) & (df_emis["Region"] == region)

    def _rows(var_mask):
        return df_emis.loc[m & var_mask, year_cols].interpolate(axis=1).values

    gfed = _rows(df_emis["Variable"].isin(_NOX_GFED_SECTORS)).sum(axis=0) * _NOX_NO2_PER_NO
    agriculture = _rows(df_emis["Variable"] == _NOX_AGRICULTURE).squeeze()
    fossil = _rows(df_emis["Variable"] == _NOX_FOSSIL_INDUSTRIAL).squeeze()
    return gfed + agriculture + fossil


def clean_rcmip_historical_for_fair(
    species,
    emissions_file=RCMIP_EMISSIONS,
    region="World",
    last_year=2014,
):
    """rcmip historical emissions as a wide, merge-ready history block up to last_year."""
    species_to_rcmip = build_species_map(species)

    df_emis = pd.read_csv(emissions_file)

    scenario = "historical"

    # keep integer year columns up to last_year, label midyear ("YYYY.5")
    hist_years = [c for c in df_emis.columns if str(c).isdigit() and int(c) <= last_year]
    out_years = [f"{int(c)}.5" for c in hist_years]

    records = []
    for specie, specie_rcmip_name in species_to_rcmip.items():
        mask = (
            (df_emis["Scenario"] == scenario)
            & (df_emis["Variable"].str.endswith("|" + specie_rcmip_name))
            & (df_emis["Region"] == region)
        )

        # guard against endswith matching a sub-sector or missing a series
        if mask.sum() != 1:
            raise MissingDataError(
                f"expected exactly 1 value for scenario={scenario}, variable "
                f"name ending with |{specie_rcmip_name}, region={region} in the "
                f"RCMIP emissions database, found {int(mask.sum())}."
            )

        # nox reconstructed from sectors (aggregate series mis-scales biomass burning)
        if specie == "NOx":
            emis_in = reconstruct_nox_emissions(df_emis, scenario, region, hist_years)
        else:
            emis_in = df_emis.loc[mask, hist_years].interpolate(axis=1).values.squeeze()

        unit = df_emis.loc[mask, "Unit"].values[0]

        record = {"scenario": scenario, "region": region, "variable": specie, "unit": unit}
        record.update(dict(zip(out_years, emis_in)))
        records.append(record)

    return pd.DataFrame(records)


def fill_from_rcmip_locally(
    fair_instance,
    emissions_file=RCMIP_EMISSIONS,
    concentration_file=RCMIP_CONCENTRATIONS,
    forcing_file=RCMIP_FORCING,
):
    """fill emissions/concentrations/forcing from rcmip scenarios using local files."""
    species_to_rcmip = build_species_map(fair_instance.species)

    df_emis = pd.read_csv(emissions_file)
    df_conc = pd.read_csv(concentration_file)
    df_forc = pd.read_csv(forcing_file)

    for scenario in fair_instance.scenarios:
        for specie, specie_rcmip_name in species_to_rcmip.items():
            if fair_instance.properties_df.loc[specie, "input_mode"] == "emissions":
                # nox reconstructed from sectors (aggregate series mis-scales biomass burning)
                if specie == "NOx":
                    year_cols = [
                        c for c in df_emis.columns
                        if str(c).isdigit() and 1750 <= int(c) <= 2500
                    ]
                    emis_in = reconstruct_nox_emissions(df_emis, scenario, "World", year_cols)
                else:
                    emis_in = (
                        df_emis.loc[
                            (df_emis["Scenario"] == scenario)
                            & (df_emis["Variable"].str.endswith("|" + specie_rcmip_name))
                            & (df_emis["Region"] == "World"),
                            "1750":"2500",
                        ]
                        .interpolate(axis=1)
                        .values.squeeze()
                    )

                if emis_in.shape[0] == 0:
                    raise MissingDataError(
                        f"I can't find a value for scenario={scenario}, variable "
                        f"name ending with {specie_rcmip_name} in the RCMIP "
                        f"emissions database."
                    )

                notnan = np.nonzero(~np.isnan(emis_in))

                # rcmip annual averages; assume midyear and interpolate to our grid
                rcmip_index = np.arange(1750.5, 2501.5)
                interpolator = interp1d(
                    rcmip_index[notnan],
                    emis_in[notnan],
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                emis = interpolator(fair_instance.timepoints)

                # fill out-of-range with nan so pre/post can be set manually
                emis[fair_instance.timepoints < 1750] = np.nan
                emis[fair_instance.timepoints > 2501] = np.nan

                # convert input unit to what fair wants
                unit = df_emis.loc[
                    (df_emis["Scenario"] == scenario)
                    & (df_emis["Variable"].str.endswith("|" + specie_rcmip_name))
                    & (df_emis["Region"] == "World"),
                    "Unit",
                ].values[0]
                emis = emis * (
                    prefix_convert[unit.split()[0]][
                        desired_emissions_units[specie].split()[0]
                    ]
                    * compound_convert[unit.split()[1].split("/")[0]][
                        desired_emissions_units[specie].split()[1].split("/")[0]
                    ]
                    * time_convert[unit.split()[1].split("/")[1]][
                        desired_emissions_units[specie].split()[1].split("/")[1]
                    ]
                )

                fill(fair_instance.emissions, emis[:, None], specie=specie, scenario=scenario)

            if fair_instance.properties_df.loc[specie, "input_mode"] == "concentration":
                conc_in = (
                    df_conc.loc[
                        (df_conc["Scenario"] == scenario)
                        & (df_conc["Variable"].str.endswith("|" + specie_rcmip_name))
                        & (df_conc["Region"] == "World"),
                        "1700":"2500",
                    ]
                    .interpolate(axis=1)
                    .values.squeeze()
                )

                if conc_in.shape[0] == 0:
                    raise MissingDataError(
                        f"I can't find a value for scenario={scenario}, variable "
                        f"name ending with {specie_rcmip_name} in the RCMIP "
                        f"concentration database."
                    )

                notnan = np.nonzero(~np.isnan(conc_in))

                # interpolate to timebounds
                rcmip_index = np.arange(1700.5, 2501.5)
                interpolator = interp1d(
                    rcmip_index[notnan],
                    conc_in[notnan],
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                conc = interpolator(fair_instance.timebounds)

                conc[fair_instance.timebounds < 1700] = np.nan
                conc[fair_instance.timebounds > 2501] = np.nan

                # convert input unit to what fair wants
                unit = df_conc.loc[
                    (df_conc["Scenario"] == scenario)
                    & (df_conc["Variable"].str.endswith("|" + specie_rcmip_name))
                    & (df_conc["Region"] == "World"),
                    "Unit",
                ].values[0]
                conc = conc * (
                    mixing_ratio_convert[unit][desired_concentration_units[specie]]
                )

                fill(
                    fair_instance.concentration,
                    conc[:, None],
                    specie=specie,
                    scenario=scenario,
                )

            if fair_instance.properties_df.loc[specie, "input_mode"] == "forcing":
                forc_in = (
                    df_forc.loc[
                        (df_forc["Scenario"] == scenario)
                        & (df_forc["Variable"].str.endswith("|" + specie_rcmip_name))
                        & (df_forc["Region"] == "World"),
                        "1750":"2500",
                    ]
                    .interpolate(axis=1)
                    .values.squeeze()
                )

                if forc_in.shape[0] == 0:
                    raise MissingDataError(
                        f"I can't find a value for scenario={scenario}, variable "
                        f"name ending with {specie_rcmip_name} in the RCMIP "
                        f"radiative forcing database."
                    )

                notnan = np.nonzero(~np.isnan(forc_in))

                # interpolate to timebounds
                rcmip_index = np.arange(1750.5, 2501.5)
                interpolator = interp1d(
                    rcmip_index[notnan],
                    forc_in[notnan],
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                forc = interpolator(fair_instance.timebounds)

                forc[fair_instance.timebounds < 1750] = np.nan
                forc[fair_instance.timebounds > 2501] = np.nan

                fill(fair_instance.forcing, forc[:, None], specie=specie, scenario=scenario)
