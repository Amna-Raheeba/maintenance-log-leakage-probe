import math

import pytest

from leakage_probe.models import (
    ALL_FEATURES,
    CLEAN_FAMILIES,
    FAMILIES,
    LEAK_FAMILIES,
    Unit,
    auc,
    family_of_feature,
    mean,
    sd,
)


def test_auc_perfect():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_auc_reversed():
    assert auc([0.8, 0.9, 0.1, 0.2], [0, 0, 1, 1]) == 0.0


def test_auc_chance_with_ties():
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5


def test_auc_partial():
    value = auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1])
    assert 0.0 < value < 1.0


def test_auc_single_class_is_nan():
    assert math.isnan(auc([0.1, 0.2], [1, 1]))


def test_mean_and_sd():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert sd([1.0, 2.0, 3.0]) == 1.0
    assert sd([5.0]) == 0.0


def test_family_partition():
    assert sorted(LEAK_FAMILIES) == ["inspections", "part_swaps", "work_orders"]
    assert sorted(CLEAN_FAMILIES) == [
        "load",
        "scheduled_maintenance",
        "thermal",
        "vibration",
    ]
    for name in LEAK_FAMILIES:
        assert FAMILIES[name]["source"] == "log"
    # The control family is a log family that does not leak - the
    # probe's precision is earned against it.
    assert FAMILIES["scheduled_maintenance"]["source"] == "log"
    assert FAMILIES["scheduled_maintenance"]["leak"] is False


def test_all_features_cover_families():
    assert len(ALL_FEATURES) == 14
    for feature in ALL_FEATURES:
        assert family_of_feature(feature) in FAMILIES


def test_family_of_feature_unknown():
    with pytest.raises(KeyError):
        family_of_feature("nope")


def test_unit_lead_and_event_days():
    unit = Unit("u1", 100, 90, {}, [(5, "inspection"), (95, "work_order")])
    assert unit.lead == 10
    assert unit.event_days(("work_order",)) == [95]
    assert unit.event_days(("inspection", "work_order")) == [5, 95]
    assert unit.to_dict()["n_events"] == 2
