"""Evaluation: seeded fleets with known leak ground truth.

Each run generates a fleet whose suspicion lead time is set by the
scenario (3, 7 or 14 days). The probe and the naive AUC-screen
baseline both try to name the leaking families; both are scored
against the constructed ground truth (the three log families leak,
the three sensor families do not). The probe additionally estimates
the lead time from the leakage curve, scored against the fleet's true
mean lead. A train/test pair of fleets prices the leak: the AUC a
benchmark would report with all features vs the AUC after removing
the families the probe flagged.
"""

from .classifier import collapse_experiment
from .features import build_cases
from .generate import generate_fleet
from .models import FAMILY_NAMES, LEAK_FAMILIES, mean
from .probe import baseline_screen, estimate_lead, probe_fleet

EVAL_SEEDS = (101, 102, 103, 104, 105)
EVAL_LEADS = (3, 7, 14)
FLEET_UNITS = 60
TRAIN_SEED_OFFSET = 500000


def run_single_seed(seed, mean_lead=7, n_units=FLEET_UNITS, n_perm=20):
    """One fleet audited end to end. Returns a result dict."""
    fleet = generate_fleet(seed, n_units, mean_lead)
    train_fleet = generate_fleet(seed + TRAIN_SEED_OFFSET, n_units, mean_lead)
    cases = build_cases(fleet)
    train_cases = build_cases(train_fleet)
    probe = probe_fleet(fleet, cases, n_perm=n_perm, seed=seed)
    baseline = baseline_screen(cases)
    flagged = [name for name in FAMILY_NAMES if probe[name]["flagged"]]
    collapse = collapse_experiment(train_cases, cases, flagged)
    true_mean_lead = mean([unit.lead for unit in fleet])
    lead_estimates = {}
    for family in LEAK_FAMILIES:
        est, _rates, _base = estimate_lead(fleet, family)
        lead_estimates[family] = est
    return {
        "seed": seed,
        "mean_lead": mean_lead,
        "true_mean_lead": true_mean_lead,
        "n_units": n_units,
        "n_cases": len(cases),
        "probe": probe,
        "baseline_flagged": sorted(baseline.keys()),
        "flagged": flagged,
        "lead_estimates": lead_estimates,
        "collapse": collapse,
    }


def _detection_counts(results, source):
    """Pooled (tp, fp, fn) over runs for 'probe' or 'baseline' flags."""
    tp = fp = fn = 0
    for result in results:
        if source == "probe":
            flagged = set(result["flagged"])
        else:
            flagged = set(result["baseline_flagged"])
        for family in FAMILY_NAMES:
            truth = family in LEAK_FAMILIES
            if family in flagged and truth:
                tp += 1
            elif family in flagged and not truth:
                fp += 1
            elif family not in flagged and truth:
                fn += 1
    return tp, fp, fn


def _precision_recall(tp, fp, fn):
    precision = tp / float(tp + fp) if tp + fp else 1.0
    recall = tp / float(tp + fn) if tp + fn else 1.0
    return precision, recall


def run_evaluation(seeds=EVAL_SEEDS, leads=EVAL_LEADS, n_perm=16):
    """The full headline evaluation over seeds x lead scenarios."""
    results = []
    for lead in leads:
        for seed in seeds:
            results.append(run_single_seed(seed, lead, n_perm=n_perm))
    probe_tp, probe_fp, probe_fn = _detection_counts(results, "probe")
    base_tp, base_fp, base_fn = _detection_counts(results, "baseline")
    probe_p, probe_r = _precision_recall(probe_tp, probe_fp, probe_fn)
    base_p, base_r = _precision_recall(base_tp, base_fp, base_fn)

    per_lead = {}
    for lead in leads:
        rows = [r for r in results if r["mean_lead"] == lead]
        lead_errors = []
        for r in rows:
            for family in LEAK_FAMILIES:
                lead_errors.append(
                    abs(r["lead_estimates"][family] - r["true_mean_lead"])
                )
        per_lead[lead] = {
            "lead_mae": mean(lead_errors),
            "mean_collapse": mean([r["collapse"]["collapse"] for r in rows]),
            "mean_all_auc": mean([r["collapse"]["all"] for r in rows]),
            "mean_cleaned_auc": mean([r["collapse"]["cleaned"] for r in rows]),
            "mean_leak_only_auc": mean([r["collapse"]["leak_only"] for r in rows]),
            "mean_sensors_only_auc": mean(
                [r["collapse"]["sensors_only"] for r in rows]
            ),
            "n_runs": len(rows),
        }
    per_family = {}
    for family in FAMILY_NAMES:
        per_family[family] = {
            "flagged_runs": sum(1 for r in results if family in r["flagged"]),
            "n_runs": len(results),
            "mean_real_auc": mean([r["probe"][family]["real_auc"] for r in results]),
            "mean_leakage_score": mean(
                [r["probe"][family]["leakage_score"] for r in results]
            ),
        }
    evaluation = {
        "seeds": list(seeds),
        "leads": list(leads),
        "n_runs": len(results),
        "probe_precision": probe_p,
        "probe_recall": probe_r,
        "baseline_precision": base_p,
        "baseline_recall": base_r,
        "per_lead": per_lead,
        "per_family": per_family,
        "results": results,
    }
    evaluation["headline"] = headline_numbers(evaluation)
    return evaluation


def headline_numbers(evaluation):
    per_lead = evaluation["per_lead"]
    lead_maes = [per_lead[lead]["lead_mae"] for lead in per_lead]
    collapses = [per_lead[lead]["mean_collapse"] for lead in per_lead]
    return {
        "probe_precision": evaluation["probe_precision"],
        "probe_recall": evaluation["probe_recall"],
        "baseline_precision": evaluation["baseline_precision"],
        "baseline_recall": evaluation["baseline_recall"],
        "mean_lead_mae": mean(lead_maes),
        "mean_collapse": mean(collapses),
        "mean_all_auc": mean(
            [per_lead[lead]["mean_all_auc"] for lead in per_lead]
        ),
        "mean_cleaned_auc": mean(
            [per_lead[lead]["mean_cleaned_auc"] for lead in per_lead]
        ),
    }
