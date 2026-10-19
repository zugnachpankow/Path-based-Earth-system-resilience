import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve paths from the repo root, not the job's cwd
from monte_carlo_runner import monte_carlo_fair, aggregate_runs

# ssp emissions/forcing come from rcmip inside the runner, so only params + species configs are passed here
fair_params_file = "data/raw/calibrated_constrained_parameters_calibration1.4.0.csv"
fair_species_configs_file = "data/raw/species_configs_properties_calibration1.4.0.csv"
time = np.arange(1750, 2300, 1)

scenarios = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']

run_files = monte_carlo_fair(
    time,
    scenarios,
    fair_params_file,
    fair_species_configs_file,
    n_runs=100,
    output_dir="output/runs_ssp"
)

aggregate_runs(run_files, "output/aggregate_ssp.nc")
