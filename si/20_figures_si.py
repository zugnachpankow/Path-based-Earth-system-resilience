"""ECS histogram: full config ensemble vs the temperature-resilient subset, per scenario.

Reads output/feedback/df_params_all.pkl; writes figures/si/si_ecs_histogram.{pdf,png}.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))

import matplotlib.transforms as mtrans
from string import ascii_lowercase
from figures_common import (
    TWO_COL, FIG_DIR, res_y, load_df_params, get_scenario_type,
    main_scenarios, composite_styles,
)

OUT_DIR = os.path.join(FIG_DIR, "si")
os.makedirs(OUT_DIR, exist_ok=True)

df_params_all = load_df_params()
print(f"Loaded df_params_all: {df_params_all.shape}", flush=True)

# ecs histogram: full ensemble vs resilient subset
HIST_SCENARIOS = [
    "REMIND-MAgPIE_3.3-4.8___Net_Zero_2050",
    "ssp126",
    "ssp534-over",
]
_styles = composite_styles(main_scenarios)   # same colours/linestyles as the combined heatmap

# common ECS bins from the full ensemble
_ecs_all = df_params_all["ecs"].dropna().values
ecs_bins = np.linspace(np.nanmin(_ecs_all), np.nanmax(_ecs_all), 31)

fig, axes = plt.subplots(1, len(HIST_SCENARIOS), figsize=(TWO_COL, TWO_COL * 0.32), sharey=True)
_lbl_off = mtrans.ScaledTranslation(9 / 72, 0, fig.dpi_scale_trans)
for k, (ax, sc) in enumerate(zip(np.atleast_1d(axes), HIST_SCENARIOS)):
    d = df_params_all[df_params_all["scenario"] == sc]
    ecs_full = d["ecs"].values                          # all members of this scenario
    ecs_res  = d.loc[d["temp"] < res_y, "ecs"].values   # resilient subset (temperature criterion)
    col, ls = _styles.get(sc, ("grey", "-"))
    ax.hist(ecs_full, bins=ecs_bins, density=True, color="grey", alpha=0.45, label="Full ensemble")
    ax.hist(ecs_res, bins=ecs_bins, density=True, histtype="step", color=col,
            linewidth=1.4, linestyle=ls, label="Resilient")
    _frac = 100.0 * (len(ecs_res) / len(d)) if len(d) else 0.0
    # Nature-style inline panel label: bold letter + italic scenario (+ % temperature-resilient)
    ax.text(0.0, 1.02, ascii_lowercase[k], transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax.text(0.0, 1.02, f"{get_scenario_type(sc)} ({_frac:.0f}% temperature-resilient)",
            transform=ax.transAxes + _lbl_off, fontsize=6, fontstyle="italic",
            va="bottom", ha="left", color="#444444", clip_on=False)
    ax.grid(True, alpha=0.3)
np.atleast_1d(axes)[0].set_ylabel("Density")
np.atleast_1d(axes)[0].legend(frameon=False, fontsize=5, loc="upper right")
fig.supxlabel("ECS [°C]", fontsize=7)           # one shared x-label
fig.tight_layout()
for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_ecs_histogram" + ext), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved si_ecs_histogram.{pdf,png}", flush=True)
