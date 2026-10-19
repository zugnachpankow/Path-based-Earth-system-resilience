"""natural-background emission floor per species, converted to each variable's csv unit."""
import pandas as pd

from fair.structure.units import desired_emissions_units
from fair_config import SPECIES_CONFIGS_NGFS
from units import emissions_unit_factor


def build_floor_map(df_emissions, species_file=SPECIES_CONFIGS_NGFS):
    """{variable: baseline_emissions in that variable's csv unit}; missing species floor at 0"""
    sp = pd.read_csv(species_file)
    base = dict(zip(sp["name"], sp["baseline_emissions"].fillna(0.0)))
    units = (
        df_emissions.drop_duplicates("variable")
        .set_index("variable")["unit"]
        .to_dict()
    )

    floors = {}
    for variable, csv_unit in units.items():
        value = float(base.get(variable, 0.0))
        if value != 0.0 and variable in desired_emissions_units:
            value *= emissions_unit_factor(desired_emissions_units[variable], csv_unit)
        floors[variable] = value
    return floors
