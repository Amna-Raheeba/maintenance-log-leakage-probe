"""Command-line interface.

Usage:
    python -m leakage_probe demo
    python -m leakage_probe evaluate --out results_dir
"""

import argparse
import csv
import json
import os
import sys

from .evaluate import headline_numbers, run_evaluation, run_single_seed
from .models import FAMILY_NAMES
from .report import render_evaluation_summary, render_single_seed_report


def _jsonable(obj):
    if isinstance(obj, dict):
        return dict((k, _jsonable(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def cmd_demo(args):
    result = run_single_seed(7)
    sys.stdout.write(render_single_seed_report(result))
    return 0


def cmd_evaluate(args):
    evaluation = run_evaluation()
    headline = headline_numbers(evaluation)
    out_dir = args.out
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    slim = {
        "seeds": evaluation["seeds"],
        "leads": evaluation["leads"],
        "n_runs": evaluation["n_runs"],
        "probe_precision": evaluation["probe_precision"],
        "probe_recall": evaluation["probe_recall"],
        "baseline_precision": evaluation["baseline_precision"],
        "baseline_recall": evaluation["baseline_recall"],
        "per_lead": evaluation["per_lead"],
        "per_family": evaluation["per_family"],
        "headline": headline,
    }
    with open(os.path.join(out_dir, "evaluation.json"), "w") as fh:
        json.dump(_jsonable(slim), fh, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "evaluation.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["family", "mean_real_auc", "mean_leakage_score", "flagged_runs", "n_runs"]
        )
        for family in FAMILY_NAMES:
            row = evaluation["per_family"][family]
            writer.writerow(
                [
                    family,
                    "%.4f" % row["mean_real_auc"],
                    "%.4f" % row["mean_leakage_score"],
                    row["flagged_runs"],
                    row["n_runs"],
                ]
            )
    with open(os.path.join(out_dir, "evaluation_summary.md"), "w") as fh:
        fh.write(render_evaluation_summary(evaluation, headline))
    with open(os.path.join(out_dir, "HEADLINE.md"), "w") as fh:
        fh.write("# Headline results\n\n")
        fh.write(
            "Event-permutation probe vs naive AUC screen, pooled over "
            "%d seeded runs (suspicion leads 3/7/14 days): probe "
            "precision %.3f / recall %.3f; baseline precision %.3f / "
            "recall %.3f. Mean lead-time estimation error %.2f days. "
            "Removing the flagged families costs %.3f AUC "
            "(%.3f -> %.3f) - the part of the benchmark score that was "
            "the crew's suspicion, not the machine's signal.\n"
            % (
                evaluation["n_runs"],
                headline["probe_precision"],
                headline["probe_recall"],
                headline["baseline_precision"],
                headline["baseline_recall"],
                headline["mean_lead_mae"],
                headline["mean_collapse"],
                headline["mean_all_auc"],
                headline["mean_cleaned_auc"],
            )
        )
    sys.stdout.write("Wrote evaluation results to %s\n" % out_dir)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="leakage_probe")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
