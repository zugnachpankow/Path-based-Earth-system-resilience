import glob
import os
from os.path import join
import xarray as xr

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from the repo root so output/ resolves
base_path = "output"
N_RUNS = 100   # per family, matches n_runs in 03/04/05


def _run_files(path):
    # assert exactly N_RUNS so a partial ensemble is never silently concatenated
    files = sorted(glob.glob(join(path, "run_*.nc")))
    if len(files) != N_RUNS:
        raise AssertionError(f"expected {N_RUNS} run_*.nc in {path}, found {len(files)}")
    return files


ngfs_path = join(base_path, "runs_ngfs")
ngfs_run_files = _run_files(ngfs_path)
ngfs_datasets = (xr.open_dataset(f) for f in ngfs_run_files)
ngfs_results = xr.concat(ngfs_datasets, dim="run")

ssp_path = join(base_path, "runs_ssp")
ssp_run_files = _run_files(ssp_path)
ssp_datasets = (xr.open_dataset(f) for f in ssp_run_files)
ssp_results = xr.concat(ssp_datasets, dim="run")

branch_off_path = join(base_path, "runs_branchoffs")
branch_off_run_files = _run_files(branch_off_path)
branch_off_datasets = (xr.open_dataset(f) for f in branch_off_run_files)
branch_off_results = xr.concat(branch_off_datasets, dim="run")

ngfs_scenarios = ngfs_results.scenario.values.tolist()
ssp_scenarios = ssp_results.scenario.values.tolist()
branch_off_scenarios = branch_off_results.scenario.values.tolist()

scenarios = ngfs_scenarios + ssp_scenarios + branch_off_scenarios

configs = ngfs_results.config.values.tolist()
runs = ngfs_results.run.values.tolist()

ngfs_temperature = ngfs_results["__xarray_dataarray_variable__"]  # (run, timebounds, scenario, config)
ssp_temperature = ssp_results["__xarray_dataarray_variable__"]
branch_off_temperature = branch_off_results["__xarray_dataarray_variable__"]

temperature = xr.concat([ngfs_temperature, ssp_temperature, branch_off_temperature], dim="scenario")

save_path = "output"
output_path = join(save_path, "all_scenarios_temperature.nc")
temperature.to_netcdf(output_path)