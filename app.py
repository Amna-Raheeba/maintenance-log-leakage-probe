"""Streamlit dashboard for the Maintenance-Log Leakage Probe.

Three tabs:
1. Probe - per-family permutation verdicts for one seeded fleet:
   real AUC, permuted AUC, leakage score, and the LEAK/clean verdict.
2. Lead-time curve - the family's pooled event rate by days before
   failure, with the estimated suspicion lead vs the true mean lead.
3. Evaluation - the aggregated multi-seed headline: probe vs the
   naive AUC screen, lead-time error, and the performance collapse.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from leakage_probe.evaluate import run_evaluation, run_single_seed
from leakage_probe.models import FAMILY_NAMES, LEAK_FAMILIES
from leakage_probe.probe import estimate_lead, leakage_curve


@st.cache_data(show_spinner=False)
def load_single_seed(seed, lead):
    return run_single_seed(seed, lead, n_units=40, n_perm=12)


@st.cache_data(show_spinner=False)
def load_evaluation():
    return run_evaluation(seeds=(101, 102), leads=(3, 7, 14), n_perm=10)


def probe_rows(result):
    rows = []
    for family in FAMILY_NAMES:
        r = result["probe"][family]
        rows.append(
            {
                "family": family,
                "truth": "leak" if family in LEAK_FAMILIES else "clean",
                "AUC": round(r["real_auc"], 3),
                "perm AUC": round(r["perm_mean_auc"], 3),
                "leakage score": round(r["leakage_score"], 3),
                "z": round(r["z"], 1),
                "verdict": "LEAK" if r["flagged"] else "clean",
            }
        )
    return rows


def tab_probe():
    st.subheader("Event-permutation probe")
    seed = st.selectbox("Fleet seed", [7, 21, 42], index=0)
    lead = st.selectbox("Suspicion lead scenario (days)", [3, 7, 14], index=1)
    result = load_single_seed(int(seed), int(lead))
    st.write(
        "True fleet mean lead: %.1f days. The probe never sees the "
        "suspicion times - it only shuffles the event timestamps "
        "within each unit and watches whose AUC collapses."
        % result["true_mean_lead"]
    )
    st.dataframe(probe_rows(result), hide_index=True)
    c = result["collapse"]
    st.write(
        "Held-out AUC - all features %.3f, probe-cleaned %.3f, "
        "sensors only %.3f, log only %.3f. Collapse: %.3f."
        % (c["all"], c["cleaned"], c["sensors_only"], c["leak_only"], c["collapse"])
    )
    st.write(
        "Naive AUC screen flagged: %s"
        % (", ".join(result["baseline_flagged"]) or "none")
    )


def tab_curve():
    st.subheader("Lead-time leakage curve")
    from leakage_probe.generate import generate_fleet

    lead = st.selectbox("Suspicion lead scenario (days)", [3, 7, 14], index=1, key="curve_lead")
    family = st.selectbox("Log family", list(LEAK_FAMILIES), index=0)
    fleet = generate_fleet(7, 60, int(lead))
    rates, baseline = leakage_curve(fleet, family)
    est, _r, _b = estimate_lead(fleet, family)
    true_mean = sum(u.lead for u in fleet) / float(len(fleet))
    st.write(
        "Pooled event rate per unit-day, by days before failure. "
        "Estimated lead %.0f days vs true mean lead %.1f days. "
        "Far-from-failure baseline rate: %.3f."
        % (est, true_mean, baseline)
    )
    st.line_chart(
        {"event rate": [rates[d] for d in range(30, -1, -1)]},
    )
    st.caption("x axis: days before failure, 30 (left) down to 0 (right).")


def tab_evaluation():
    st.subheader("Headline evaluation (2 seeds x 3 lead scenarios)")
    evaluation = load_evaluation()
    headline = evaluation["headline"]
    st.write(
        "Probe: precision %.3f, recall %.3f. Naive AUC screen: "
        "precision %.3f, recall %.3f."
        % (
            headline["probe_precision"],
            headline["probe_recall"],
            headline["baseline_precision"],
            headline["baseline_recall"],
        )
    )
    rows = []
    for lead in evaluation["leads"]:
        row = evaluation["per_lead"][lead]
        rows.append(
            {
                "true lead": lead,
                "lead MAE": round(row["lead_mae"], 2),
                "AUC all": round(row["mean_all_auc"], 3),
                "AUC cleaned": round(row["mean_cleaned_auc"], 3),
                "collapse": round(row["mean_collapse"], 3),
            }
        )
    st.dataframe(rows, hide_index=True)
    st.write("Mean lead-time error: %.2f days." % headline["mean_lead_mae"])


def main():
    st.title("Maintenance-Log Leakage Probe")
    st.write(
        "Does the feature predict the failure - or does it record "
        "that the crew already suspected it? An event-permutation "
        "null over the maintenance log tells the two apart."
    )
    tab1, tab2, tab3 = st.tabs(["Probe", "Lead-time curve", "Evaluation"])
    with tab1:
        tab_probe()
    with tab2:
        tab_curve()
    with tab3:
        tab_evaluation()


if __name__ == "__main__":
    main()
