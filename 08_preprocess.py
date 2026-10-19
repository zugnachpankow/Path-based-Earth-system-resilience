"""cache the 20-yr running-mean gmst per scenario; in: concatenated temperature; out: running_mean_temps.pkl"""
import os
import sys
import pickle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve paths from the repo root, not the job's cwd
from resilience import load_temperature, compute_running_means

TEMPERATURE_FILE = "output/all_scenarios_temperature.nc"
RUNNING_MEANS_FILE = "output/running_mean_temps.pkl"

temperature = load_temperature(TEMPERATURE_FILE)
scenarios = temperature.scenario.values.tolist()

running_means = compute_running_means(temperature, scenarios)

with open(RUNNING_MEANS_FILE, "wb") as f:
    pickle.dump(running_means, f)
