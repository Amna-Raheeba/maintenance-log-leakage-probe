import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "evaluation.json")


def test_stored_results_exist_and_parse():
    assert os.path.isfile(RESULTS)
    with open(RESULTS) as fh:
        payload = json.load(fh)
    for key in ("seeds", "leads", "n_runs", "per_family", "per_lead", "headline"):
        assert key in payload, key


def test_stored_headline_keys():
    with open(RESULTS) as fh:
        payload = json.load(fh)
    headline = payload["headline"]
    for key in (
        "probe_precision",
        "probe_recall",
        "baseline_precision",
        "baseline_recall",
        "mean_lead_mae",
        "mean_collapse",
        "mean_all_auc",
        "mean_cleaned_auc",
    ):
        assert key in headline, key


def test_stored_headline_values_sane():
    with open(RESULTS) as fh:
        payload = json.load(fh)
    headline = payload["headline"]
    # The probe never cried wolf in the stored evaluation...
    assert headline["probe_precision"] >= 0.99
    # ...but it honestly misses the faintest leaks (3-day suspicion
    # inside a 14-day horizon), so pooled recall sits well below 1.
    assert 0.5 < headline["probe_recall"] < 1.0
    assert headline["baseline_precision"] < headline["probe_precision"]
    assert headline["baseline_recall"] < headline["probe_recall"]
    assert headline["mean_lead_mae"] < 4.0
    assert headline["mean_collapse"] > 0.0
