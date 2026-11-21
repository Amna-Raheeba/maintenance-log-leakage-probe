"""Notebook 02 - the headline evaluation experiment.

Runs the full seeded evaluation (5 seeds x suspicion leads 3/7/14
days) and prints the summary: probe vs baseline detection, per-family
scores, lead-time estimation error, and the performance collapse per
lead scenario. Runnable as a plain script:
python notebooks/02_evaluation_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leakage_probe.evaluate import headline_numbers, run_evaluation
from leakage_probe.report import render_evaluation_summary


def main():
    evaluation = run_evaluation()
    print(render_evaluation_summary(evaluation, headline_numbers(evaluation)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
