"""Notebook 01 - demo run.

Runs the probe on a single demo fleet (seed 7, suspicion lead
scenario 7 days) and prints the per-family verdict table, the
lead-time estimates, and the collapse experiment. Runnable as a
plain script: python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leakage_probe.evaluate import run_single_seed
from leakage_probe.report import render_single_seed_report


def main():
    result = run_single_seed(7)
    print(render_single_seed_report(result))
    truth = {"work_orders", "inspections", "part_swaps"}
    print(
        "Probe verdict exact: %s (flagged %s)"
        % (set(result["flagged"]) == truth, ", ".join(result["flagged"]))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
