"""resilience-parameter sensitivity (sobol + lhs monte carlo) for three focal ssp scenarios; writes output/sensitivity"""
import os
import sys
import pickle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve output/ from the repo root
from resilience import load_tipping
from sensitivity import run_sobol_mc, problem

RUNNING_MEANS_FILE = "output/running_mean_temps.pkl"
TIPPING_DIR = "output/tipping"
# rate-range-specific subdir so different rate cutoffs don't overwrite each other
_rlo, _rhi = problem["bounds"][2]
save_path = f"output/sensitivity/rate{_rlo}_{_rhi}/"
os.makedirs(save_path, exist_ok=True)
print(f"Saving sensitivity to {save_path} (rate range {_rlo}-{_rhi} °C/yr)", flush=True)

FOCAL_SCENARIOS = ["ssp126", "ssp245", "ssp534-over"]

with open(RUNNING_MEANS_FILE, "rb") as f:
    running_mean_temps = pickle.load(f)
running_mean_temps = {
    k.replace("(", "").replace(")", ""): v for k, v in running_mean_temps.items()
}
tipping_data_dict = load_tipping(TIPPING_DIR)

for sc in FOCAL_SCENARIOS:
    print(f"\n=== {sc} ===")
    rm = running_mean_temps[sc]
    tip_prob = tipping_data_dict[sc]["prob_any_tipping"]
    run_sobol_mc(rm, tip_prob, save_path, sc)

print("\nDone.")
