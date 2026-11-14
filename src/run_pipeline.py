"""
Entry point. Runs the full pipeline end to end.

    python src/run_pipeline.py

Steps 0-5 execute on import (data, best_order, final_fit are created at
module level in steps0to5.py). Steps 6-9 then pick those objects up.
"""

import os

# Make sure output directories exist before any script tries to save a figure
for d in ("data/raw", "results", "notebooks"):
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("STEPS 0-5: data, stationarity, ARMA selection and fit")
print("=" * 60)

from steps0to5 import data, best_order, final_fit

print("\n" + "=" * 60)
print("STEPS 6-9: change-point detection and validation")
print("=" * 60)

# changepoint_pipeline.py expects `data`, `best_order` and `final_fit` to
# exist in its namespace. Injecting them into builtins is the least
# invasive way to share them without restructuring either script into
# functions — fine for a research pipeline, though if you later refactor
# into proper functions, pass them as arguments instead.
import builtins

builtins.data = data
builtins.best_order = best_order
builtins.final_fit = final_fit

import changepoint_pipeline 
 # noqa: F401  (runs on import)

print("\n" + "=" * 60)
print("Pipeline complete. See results/ for figures and CSVs.")
print("=" * 60)
