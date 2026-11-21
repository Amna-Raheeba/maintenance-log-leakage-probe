from leakage_probe.evaluate import (
    _detection_counts,
    headline_numbers,
    run_evaluation,
    run_single_seed,
)
from leakage_probe.models import LEAK_FAMILIES


def test_run_single_seed_structure():
    result = run_single_seed(3, 7, n_units=30, n_perm=8)
    assert result["n_units"] == 30
    assert result["n_cases"] > 500
    assert set(result["flagged"]) == set(LEAK_FAMILIES)
    for family in LEAK_FAMILIES:
        assert family in result["lead_estimates"]


def test_run_single_seed_lead_estimate_sane():
    result = run_single_seed(3, 7, n_units=30, n_perm=8)
    for est in result["lead_estimates"].values():
        assert 0.0 <= est <= 30.0


def test_detection_counts_perfect_probe():
    fake = [
        {
            "flagged": list(LEAK_FAMILIES),
            "baseline_flagged": ["vibration"],
        }
    ]
    tp, fp, fn = _detection_counts(fake, "probe")
    assert (tp, fp, fn) == (3, 0, 0)
    tp, fp, fn = _detection_counts(fake, "baseline")
    assert (tp, fp, fn) == (0, 1, 3)


def test_run_evaluation_small():
    evaluation = run_evaluation(seeds=(201,), leads=(7,), n_perm=8)
    assert evaluation["n_runs"] == 1
    headline = headline_numbers(evaluation)
    for key in (
        "probe_precision",
        "probe_recall",
        "baseline_precision",
        "baseline_recall",
        "mean_lead_mae",
        "mean_collapse",
    ):
        assert key in headline
    assert headline["probe_precision"] >= 0.9
    assert headline["probe_recall"] >= 0.9
    assert headline["mean_lead_mae"] < 4.0


def test_run_single_seed_control_family_clean():
    result = run_single_seed(3, 7, n_units=30, n_perm=8)
    assert "scheduled_maintenance" not in result["flagged"]
    assert "scheduled_maintenance" not in LEAK_FAMILIES
