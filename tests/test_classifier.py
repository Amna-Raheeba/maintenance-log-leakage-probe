from leakage_probe.classifier import (
    LogisticRegression,
    collapse_experiment,
    features_of_families,
)
from leakage_probe.features import build_cases
from leakage_probe.generate import generate_fleet
from leakage_probe.models import ALL_FEATURES, Case


def _toy_cases():
    cases = []
    for i in range(60):
        y = 1 if i >= 30 else 0
        feats = dict((f, 0.0) for f in ALL_FEATURES)
        feats["vib_level"] = 5.0 * y + (i % 5) * 0.1
        cases.append(Case("toy", i, y, feats))
    return cases


def test_logistic_regression_learns_separable():
    cases = _toy_cases()
    model = LogisticRegression(["vib_level"], iterations=300).fit(cases)
    assert model.score_auc(cases) > 0.95


def test_logistic_regression_deterministic():
    cases = _toy_cases()
    a = LogisticRegression(["vib_level"], iterations=100).fit(cases)
    b = LogisticRegression(["vib_level"], iterations=100).fit(cases)
    assert a.weights == b.weights
    assert a.bias == b.bias


def test_features_of_families():
    feats = features_of_families(["work_orders", "vibration"])
    assert sorted(feats) == ["vib_level", "vib_slope7", "wo_count7", "wo_emergency7"]


def test_collapse_experiment_keys_and_direction():
    train = build_cases(generate_fleet(31, 30, 7))
    test = build_cases(generate_fleet(32, 30, 7))
    out = collapse_experiment(train, test, ["work_orders", "inspections", "part_swaps"])
    for key in ("all", "cleaned", "sensors_only", "leak_only", "collapse"):
        assert key in out
    assert out["all"] > 0.8
    assert out["collapse"] >= -0.01
    assert out["leak_only"] > 0.6


def test_collapse_with_no_flags_is_zero():
    train = build_cases(generate_fleet(31, 30, 7))
    test = build_cases(generate_fleet(32, 30, 7))
    out = collapse_experiment(train, test, [])
    assert abs(out["collapse"]) < 1e-9
