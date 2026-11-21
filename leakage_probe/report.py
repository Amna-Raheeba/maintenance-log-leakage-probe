"""Plain-text and Markdown rendering of probe and evaluation results."""

from .models import FAMILY_NAMES, LEAK_FAMILIES


def _verdict(result):
    return "LEAK" if result["flagged"] else "clean"


def render_single_seed_report(result):
    lines = []
    lines.append(
        "Maintenance-log leakage probe - seed %d, suspicion lead "
        "scenario %d days (true fleet mean lead %.1f days)"
        % (result["seed"], result["mean_lead"], result["true_mean_lead"])
    )
    lines.append(
        "Fleet: %d units, %d prediction cases (fail within 14 days = 1)"
        % (result["n_units"], result["n_cases"])
    )
    lines.append("")
    lines.append(
        "%-12s %-6s %7s %7s %7s %6s  %s"
        % ("family", "truth", "AUC", "permAUC", "leak", "z", "verdict")
    )
    for family in FAMILY_NAMES:
        r = result["probe"][family]
        truth = "leak" if family in LEAK_FAMILIES else "clean"
        lines.append(
            "%-12s %-6s %7.3f %7.3f %7.3f %6.1f  %s"
            % (
                family,
                truth,
                r["real_auc"],
                r["perm_mean_auc"],
                r["leakage_score"],
                r["z"],
                _verdict(r),
            )
        )
    lines.append("")
    lines.append(
        "Naive AUC screen flagged: %s"
        % (", ".join(result["baseline_flagged"]) or "none")
    )
    lines.append("Probe flagged: %s" % (", ".join(result["flagged"]) or "none"))
    lines.append("")
    lines.append("Lead-time estimates (days before failure the leak switches on):")
    for family in LEAK_FAMILIES:
        lines.append(
            "  %-12s estimated %.1f  (true mean %.1f)"
            % (family, result["lead_estimates"][family], result["true_mean_lead"])
        )
    c = result["collapse"]
    lines.append("")
    lines.append(
        "Held-out AUC: all features %.3f | probe-cleaned %.3f | "
        "sensors only %.3f | log only %.3f"
        % (c["all"], c["cleaned"], c["sensors_only"], c["leak_only"])
    )
    lines.append(
        "Collapse when the flagged families are removed: %.3f AUC"
        % c["collapse"]
    )
    return "\n".join(lines) + "\n"


def render_evaluation_summary(evaluation, headline):
    lines = []
    lines.append("# Evaluation summary")
    lines.append("")
    lines.append(
        "Seeded fleets: %d runs (%d seeds x suspicion leads %s), "
        "60 units per fleet. Ground truth: three maintenance-log "
        "families leak (work orders, inspections, part swaps); the "
        "three sensor families and the scheduled-maintenance control "
        "family are clean."
        % (
            evaluation["n_runs"],
            len(evaluation["seeds"]),
            ", ".join(str(x) for x in evaluation["leads"]),
        )
    )
    lines.append("")
    lines.append("## Detection: probe vs naive AUC screen")
    lines.append("")
    lines.append(
        "- Probe (event-permutation null): precision %.3f, recall %.3f"
        % (headline["probe_precision"], headline["probe_recall"])
    )
    lines.append(
        "- Baseline (flag any family with AUC >= 0.70): precision %.3f, "
        "recall %.3f"
        % (headline["baseline_precision"], headline["baseline_recall"])
    )
    lines.append("")
    lines.append("## Per family (pooled over all runs)")
    lines.append("")
    lines.append("| family | truth | mean AUC | mean leakage score | flagged (runs) |")
    lines.append("|---|---|---|---|---|")
    for family in FAMILY_NAMES:
        row = evaluation["per_family"][family]
        truth = "leak" if family in LEAK_FAMILIES else "clean"
        lines.append(
            "| %s | %s | %.3f | %.3f | %d/%d |"
            % (
                family,
                truth,
                row["mean_real_auc"],
                row["mean_leakage_score"],
                row["flagged_runs"],
                row["n_runs"],
            )
        )
    lines.append("")
    lines.append("## Lead-time estimation and collapse, per suspicion lead")
    lines.append("")
    lines.append("| true lead (days) | lead MAE | AUC all | AUC cleaned | collapse | AUC log only | AUC sensors only |")
    lines.append("|---|---|---|---|---|---|---|")
    for lead in evaluation["leads"]:
        row = evaluation["per_lead"][lead]
        lines.append(
            "| %d | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f |"
            % (
                lead,
                row["lead_mae"],
                row["mean_all_auc"],
                row["mean_cleaned_auc"],
                row["mean_collapse"],
                row["mean_leak_only_auc"],
                row["mean_sensors_only_auc"],
            )
        )
    lines.append("")
    lines.append(
        "Mean lead-time estimation error: %.2f days. Mean collapse: "
        "%.3f AUC (all-features %.3f -> cleaned %.3f)."
        % (
            headline["mean_lead_mae"],
            headline["mean_collapse"],
            headline["mean_all_auc"],
            headline["mean_cleaned_auc"],
        )
    )
    return "\n".join(lines) + "\n"
