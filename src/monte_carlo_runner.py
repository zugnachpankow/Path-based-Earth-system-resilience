"""run monte carlo FAIR with the stochastic climate model; emissions/forcing in, per-run temperature nc out."""
import os
import copy

import xarray as xr
from tqdm import trange

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

from rcmip import fill_from_rcmip_locally
from fair_config import (
    load_fair_params,
    fair_override_file,
    natural_forcing,
    solar_trend_shape,
)


def monte_carlo_fair(
    time,
    scenarios,
    params_file,
    species_configs_file,
    emissions_file=None,
    forcing_file=None,
    n_runs=100,
    seed_start=1234,
    output_dir="runs",
    stochastic=True
):
    """run monte carlo FAIR and write one temperature nc per run under output_dir"""
    os.makedirs(output_dir, exist_ok=True)
    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(time[0], time[-1], 1)
    f.define_scenarios(scenarios)
    fair_params_file = params_file
    # df_fair holds fair config columns; solar holds the per-config solar amplitude/trend
    df_fair, solar = load_fair_params(fair_params_file)
    configs = df_fair.index
    f.define_configs(configs)
    fair_species_configs_file = species_configs_file
    species, properties = read_properties(filename=fair_species_configs_file)
    f.define_species(species, properties)
    f.allocate()
    if (emissions_file is None) != (forcing_file is None):
        raise ValueError(
            "emissions_file and forcing_file must be provided together (or both "
            "left as None to fall back to RCMIP); got "
            f"emissions_file={emissions_file!r}, forcing_file={forcing_file!r}."
        )
    if emissions_file is not None:
        f.fill_from_csv(
            emissions_file=emissions_file,
            forcing_file=forcing_file
            )
    else:
        fill_from_rcmip_locally(f)
    # natural forcing scaled per config, shared by every scenario family; replaces any
    # solar/volcanic forcing from the source
    # forcing_scale has no effect on prescribed-forcing species, so this pre-scaling
    # is the only scaling that applies
    volcanic, solar_erf = natural_forcing(f.timebounds)
    trend = solar_trend_shape(f.timebounds)
    fill(
        f.forcing,
        volcanic[:, None, None] * df_fair["forcing_scale[Volcanic]"].values.squeeze(),
        specie="Volcanic",
    )
    if solar is not None:
        fill(
            f.forcing,
            solar_erf[:, None, None] * solar["fscale_solar_amplitude"].values.squeeze()
            + trend[:, None, None] * solar["fscale_solar_trend"].values.squeeze(),
            specie="Solar",
        )
    f.fill_species_configs(fair_species_configs_file)
    with fair_override_file(df_fair) as override_file:
        f.override_defaults(override_file)
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)
    f_baseline = copy.deepcopy(f)
    files = []
    for i in trange(n_runs, desc="Monte Carlo"):
        f = copy.deepcopy(f_baseline)
        seed = seed_start + i*99          # shared fair seeds, fixed across runs
        for config in configs:
            fill(f.climate_configs['stochastic_run'], stochastic, config=config)
            fill(f.climate_configs['use_seed'], True, config=config)
            fill(f.climate_configs['seed'], seed, config=config)

        f.run(progress=False)
        result = f.temperature.sel(layer=0)
        out_file = os.path.join(output_dir, f"run_{i:04d}.nc")
        result.to_netcdf(out_file)
        files.append(out_file)

    return files


def aggregate_runs(run_files, out_path):
    """concatenate per-run temperature files along run, write mean/std/percentiles to out_path"""
    da = xr.concat([xr.open_dataarray(f) for f in run_files], dim="run")
    quantiles = {"p0": 0.0, "p5": 0.05, "p16": 0.16, "p84": 0.84, "p95": 0.95, "p100": 1.0}
    stats = {"mean": da.mean("run"), "std": da.std("run")}
    for name, q in quantiles.items():
        stats[name] = da.quantile(q, dim="run").drop_vars("quantile")
    xr.Dataset(stats).to_netcdf(out_path)
    return out_path