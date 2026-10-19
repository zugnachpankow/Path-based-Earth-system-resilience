"""per-scenario tipping runner; input: all_scenarios_temperature.nc; output: <scenario>_tipping_probabilities.nc"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from tipping import compute_tip_prob
from tipping_params import sample_lhs_params
from temperature import processed_temperature

OUTPUT = Path(__file__).resolve().parents[1] / "output"

if __name__ == "__main__":
    path = str(OUTPUT / "all_scenarios_temperature.nc")
    temperature = xr.open_dataset(path)

    # drop brackets so the scenario name can be passed via .sh
    temperature = temperature.assign_coords(
        scenario=[s.replace("(", "").replace(")", "") for s in temperature.scenario.values]
    )

    configs = temperature.config.values.tolist()
    runs = temperature.run.values.tolist()

    scenario = np.array(sys.argv[1:], dtype=str)[0]

    n_samples = 1000
    params = sample_lhs_params(n_samples=n_samples)

    # tipping forcing uses the same processed series as the temperature criterion; compute_tip_prob
    # holds the last valid value to T_END
    forcing = processed_temperature(
        temperature["__xarray_dataarray_variable__"].sel(scenario=scenario)
    )
    prob_any, prob_any_sample, prob_elements = compute_tip_prob(
        forcing, params, n_samples=n_samples, configs=configs, runs=runs,
    )

    ds_prob = xr.Dataset(
        {
            "prob_any_tipping": prob_any,
            "prob_any_tipping_sample": prob_any_sample,
            "prob_element_tipping": prob_elements
        }
    )

    out_dir = OUTPUT / "tipping"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_prob.to_netcdf(str(out_dir / f"{scenario}_tipping_probabilities.nc"))

    # solver diagnostics appended to solver_report.csv; the next step refuses to proceed
    # if any n_solver_failures > 0
    import csv
    n_fail = int(prob_any.attrs.get("n_solver_failures", 0))
    n_disagree = int(prob_any.attrs.get("n_tip_final_disagreements", 0))
    print(f"  solver: n_solver_failures={n_fail}  n_tip_final_disagreements={n_disagree}",
          flush=True)
    report = out_dir / "solver_report.csv"
    write_header = not report.exists()
    with open(report, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["scenario", "n_solver_failures", "n_tip_final_disagreements"])
        w.writerow([scenario, n_fail, n_disagree])
