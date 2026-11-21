from leakage_probe.evaluate import run_evaluation, run_single_seed
from leakage_probe.report import (
    render_evaluation_summary,
    render_single_seed_report,
)


def test_single_seed_report_contents():
    result = run_single_seed(3, 7, n_units=30, n_perm=8)
    text = render_single_seed_report(result)
    assert "LEAK" in text
    assert "work_orders" in text
    assert "Collapse" in text
    assert "Naive AUC screen flagged" in text


def test_evaluation_summary_contents():
    evaluation = run_evaluation(seeds=(201,), leads=(7,), n_perm=8)
    text = render_evaluation_summary(evaluation, evaluation["headline"])
    assert "precision" in text
    assert "part_swaps" in text
    assert "Lead-time" in text
