"""resilience-parameter sensitivity; in: running means + tipping probs; out: sobol/mc pickles."""
import os
import pickle

import numpy as np
from tqdm import tqdm
from SALib.sample import saltelli
from SALib.analyze import sobol
from pyDOE import lhs

SEED = 1234   # reproducible saltelli + lhs sampling

problem = {
    'num_vars': 4,
    'names': ['res_x', 'res_y', 'rate', 'tipping'],
    'bounds': [
        [2080, 2150],
        [1.3, 2.0],
        [0.025, 0.055],
        [0.0, 1.0],
    ]
}


def resilience_index(rm, res_x, res_y, rate, tipping_switch, tip_prob):
    climate_violation = (
        rm.sel(timebounds=slice(res_x, None)) > res_y
    ).any(dim="timebounds")

    rate_of_change = rm.diff("timebounds").rolling(timebounds=10, center=True).mean()
    rate_violation = (
        rate_of_change.sel(timebounds=slice(2000, 2096)) > rate
    ).any(dim="timebounds")

    binary_resilient = ~(climate_violation | rate_violation)

    # apply tipping constraint only if switch > 0.5
    if tipping_switch > 0.5:
        res_s = binary_resilient.set_index(member=["config", "run"]).unstack("member")
        resilience = res_s.astype(float) * (1 - tip_prob)
        # nan-aware mean: failed tipping solves are nan and skipped
        resilience_val = float(resilience.mean())
    else:
        resilience_val = float(binary_resilient.mean())

    return resilience_val


def run_sobol_mc(rm, tip_prob, save_path, sc, N=1024, N_mc=10000):
    # nan tipping solves stay nan and are dropped by the nan-aware averaging
    tip_prob_sc = tip_prob
    sobol_path  = save_path + f"sobol_{sc}.pkl"
    mc_path     = save_path + f"mc_{sc}.pkl"

    if os.path.exists(sobol_path):
        with open(sobol_path, "rb") as f:
            d = pickle.load(f)
        param_values, Y, Si = d['param_values'], d['Y'], d['Si']
    else:
        # salib 1.5.2 saltelli.sample has no seed kwarg (deterministic anyway)
        param_values = saltelli.sample(problem, N, calc_second_order=True)
        Y = np.array([
            resilience_index(rm, int(p[0]), p[1], p[2], p[3], tip_prob_sc)
            for p in tqdm(param_values, desc=f"Sobol {sc}")
        ])
        Si = sobol.analyze(problem, Y, calc_second_order=True, seed=SEED, print_to_console=True)
        with open(sobol_path, "wb") as f:
            pickle.dump({'param_values': param_values, 'Y': Y, 'Si': Si}, f)

    if os.path.exists(mc_path):
        with open(mc_path, "rb") as f:
            d = pickle.load(f)
        param_mc, Y_mc = d['param_mc'], d['Y_mc']
        baseline_no_tip, baseline_tip = d['baseline_no_tip'], d['baseline_tip']
    else:
        np.random.seed(SEED)   # pydoe lhs draws from numpy global rng
        lhs_samples = lhs(4, samples=N_mc, criterion='maximin')
        param_mc = np.zeros_like(lhs_samples)
        for i, (lo, hi) in enumerate(problem['bounds']):
            param_mc[:, i] = lo + lhs_samples[:, i] * (hi - lo)
        Y_mc = np.array([
            resilience_index(rm, int(p[0]), p[1], p[2], p[3], tip_prob_sc)
            for p in tqdm(param_mc, desc=f"MC {sc}")
        ])
        baseline_no_tip = resilience_index(rm, 2100, 1.5, 0.04, 0.0, tip_prob_sc)
        baseline_tip    = resilience_index(rm, 2100, 1.5, 0.04, 1.0, tip_prob_sc)
        with open(mc_path, "wb") as f:
            pickle.dump({'param_mc': param_mc, 'Y_mc': Y_mc,
                         'baseline_no_tip': baseline_no_tip, 'baseline_tip': baseline_tip}, f)

    return Si, param_values, Y, param_mc, Y_mc, baseline_no_tip, baseline_tip
