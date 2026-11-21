"""The probe: an event-permutation null plus a lead-time curve.

For each feature family the probe asks one question: does this
family's predictive power depend on the *alignment* between the
maintenance-event timeline and the failure time? Shuffle the event
days within each unit - same events, same counts, alignment destroyed
- and re-measure. Genuine sensor signal is untouched by the shuffle
and survives. Intervention leakage lives entirely in the alignment
and collapses.

The family's scoring rule (its strongest single feature, or the
standardised aggregate of its features) is chosen once on the real
alignment, then held fixed across permutations, so the comparison is
apples-to-apples.

The lead-time curve pools the family's daily event rate by days
before failure. For a leaking family the curve is a step: background
rate far from failure, elevated rate inside the suspicion window. The
half-rise crossing of that step estimates the suspicion lead time
without ever seeing the ground-truth suspicion times.
"""

import random

from .features import extract_features, permuted_event_index, unit_lookup
from .models import FAMILIES, FAMILY_NAMES, auc, mean, sd

# Flagging rule for the permutation test.
FLAG_MIN_SCORE = 0.05
FLAG_MIN_REAL_AUC = 0.58
FLAG_MIN_Z = 2.5

# Baseline screener: the naive practice the probe is compared against -
# flag any family whose univariate AUC looks "too good to be true".
BASELINE_AUC_THRESHOLD = 0.70

CURVE_MAX_LEAD = 30
BASELINE_LEAD_LO = 40
BASELINE_LEAD_HI = 60


def _candidate_rules(cases, family):
    """{rule name: per-case values} on the real alignment."""
    features = FAMILIES[family]["features"]
    labels = [case.label for case in cases]
    rules = {}
    for feature in features:
        rules[feature] = [case.features[feature] for case in cases]
    agg = [0.0] * len(cases)
    stats = {}
    for feature in features:
        values = rules[feature]
        mu = mean(values)
        scale = sd(values)
        if scale == 0.0:
            scale = 1.0
        stats[feature] = (mu, scale)
        for i, v in enumerate(values):
            agg[i] += (v - mu) / scale
    rules["__aggregate__"] = agg
    return rules, stats, labels


def _rule_values(cases, lookup, family, rule, stats, event_indexes):
    """Per-case values of a fixed rule under alternative event indexes."""
    features = FAMILIES[family]["features"]
    values = []
    if rule == "__aggregate__":
        for case in cases:
            feats = extract_features(
                lookup[case.unit_id], case.day, event_indexes[case.unit_id]
            )
            total = 0.0
            for feature in features:
                mu, scale = stats[feature]
                total += (feats[feature] - mu) / scale
            values.append(total)
    else:
        for case in cases:
            feats = extract_features(
                lookup[case.unit_id], case.day, event_indexes[case.unit_id]
            )
            values.append(feats[rule])
    return values


def probe_family(fleet, cases, family, n_perm=20, seed=99):
    """Permutation test for one family. Returns a result dict."""
    rules, stats, labels = _candidate_rules(cases, family)
    best_rule = None
    best_auc = -1.0
    for name, values in rules.items():
        value = auc(values, labels)
        if value == value and value > best_auc:
            best_auc = value
            best_rule = name
    result = {
        "family": family,
        "rule": best_rule,
        "real_auc": best_auc,
        "source": FAMILIES[family]["source"],
        "truth_leak": FAMILIES[family]["leak"],
    }
    if FAMILIES[family]["source"] == "sensor":
        # The permutation never touches sensor series, so the null
        # distribution is a point mass at the real AUC.
        result.update(
            {
                "perm_mean_auc": best_auc,
                "perm_sd_auc": 0.0,
                "leakage_score": 0.0,
                "z": 0.0,
                "flagged": False,
            }
        )
        return result
    lookup = unit_lookup(fleet)
    rng = random.Random(seed)
    perm_aucs = []
    for _p in range(n_perm):
        indexes = dict(
            (unit.unit_id, permuted_event_index(unit, rng)) for unit in fleet
        )
        values = _rule_values(cases, lookup, family, best_rule, stats, indexes)
        perm_aucs.append(auc(values, labels))
    perm_mean = mean(perm_aucs)
    perm_sd = sd(perm_aucs)
    score = best_auc - perm_mean
    if perm_sd > 0:
        z = score / perm_sd
    else:
        z = 99.0 if score > 0 else 0.0
    flagged = (
        score >= FLAG_MIN_SCORE and best_auc >= FLAG_MIN_REAL_AUC and z >= FLAG_MIN_Z
    )
    result.update(
        {
            "perm_mean_auc": perm_mean,
            "perm_sd_auc": perm_sd,
            "leakage_score": score,
            "z": z,
            "flagged": flagged,
        }
    )
    return result


def probe_fleet(fleet, cases, n_perm=20, seed=99):
    """Probe every family; returns {family: result dict}."""
    results = {}
    for offset, family in enumerate(FAMILY_NAMES):
        results[family] = probe_family(
            fleet, cases, family, n_perm=n_perm, seed=seed + offset
        )
    return results


def baseline_screen(cases):
    """The naive AUC screen: flag families that look too predictive.

    Returns {family: real AUC} for flagged families only. This screen
    cannot tell a strong sensor from a leak - both are just predictive
    - which is exactly the failure mode the probe exists to fix.
    """
    flagged = {}
    for family in FAMILY_NAMES:
        rules, _stats, labels = _candidate_rules(cases, family)
        best = -1.0
        for values in rules.values():
            value = auc(values, labels)
            if value == value and value > best:
                best = value
        if best >= BASELINE_AUC_THRESHOLD:
            flagged[family] = best
    return flagged


def leakage_curve(fleet, family, max_lead=CURVE_MAX_LEAD):
    """Pooled daily event rate by days before failure.

    Returns (rates, baseline) where rates[d] is the mean number of the
    family's events per unit-day exactly d days before failure, for
    d in 0..max_lead, and baseline is the same rate measured far from
    failure (40-60 days out). Sensor families have no events and
    return zeros.
    """
    kinds = FAMILIES[family]["event_kinds"]
    counts = [0] * (max_lead + 1)
    far_count = 0
    for unit in fleet:
        for day, kind in unit.events:
            if kind not in kinds:
                continue
            d = unit.failure_time - day
            if 0 <= d <= max_lead:
                counts[d] += 1
            elif BASELINE_LEAD_LO <= d <= BASELINE_LEAD_HI:
                far_count += 1
    n_units = len(fleet)
    rates = [c / float(n_units) for c in counts]
    far_span = BASELINE_LEAD_HI - BASELINE_LEAD_LO + 1
    baseline = far_count / float(n_units * far_span) if n_units else 0.0
    return rates, baseline


def _smooth(values, width=3):
    out = []
    half = width // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(mean(values[lo:hi]))
    return out


def estimate_lead(fleet, family):
    """Estimate the suspicion lead time from the leakage curve.

    The half-rise crossing: the largest days-before-failure at which
    the smoothed event rate crosses halfway from the far-from-failure
    baseline to its near-failure peak. Returns (estimate, rates,
    baseline). An estimate of 0 means no step was found.
    """
    rates, baseline = leakage_curve(fleet, family)
    smooth = _smooth(rates)
    peak = mean(smooth[0:3])
    if peak <= baseline:
        return 0.0, rates, baseline
    threshold = baseline + 0.5 * (peak - baseline)
    # Rates decay as d grows (further from failure), so the largest d
    # still above the half-rise threshold is the step's outer edge.
    estimate = 0
    for d in range(len(smooth)):
        if smooth[d] >= threshold:
            estimate = d
    return float(estimate), rates, baseline
