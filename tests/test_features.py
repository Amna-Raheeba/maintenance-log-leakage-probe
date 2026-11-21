import random

from leakage_probe.features import (
    build_cases,
    event_index_of,
    extract_features,
    permuted_event_index,
)
from leakage_probe.models import ALL_FEATURES, HORIZON, Unit


def _hand_unit():
    sensors = {
        "vib": [float(t) for t in range(100)],
        "temp": [50.0] * 100,
        "load": [60.0] * 100,
        "ambient": [22.0] * 100,
    }
    events = [
        (10, "work_order"),
        (12, "work_order"),
        (15, "emergency_order"),
        (30, "inspection"),
        (33, "scheduled_service"),
        (40, "part_swap"),
    ]
    return Unit("hand", 100, 90, sensors, events)


def test_extract_feature_keys():
    feats = extract_features(_hand_unit(), 20)
    assert sorted(feats.keys()) == sorted(ALL_FEATURES)


def test_window_counts_hand_unit():
    feats = extract_features(_hand_unit(), 20)
    # Window (13, 20]: no work orders (10, 12 are outside), none.
    assert feats["wo_count7"] == 0.0
    feats = extract_features(_hand_unit(), 16)
    # Window (9, 16]: work orders at 10, 12; emergency at 15.
    assert feats["wo_count7"] == 2.0
    assert feats["wo_emergency7"] == 1.0


def test_long_window_counts():
    feats = extract_features(_hand_unit(), 45)
    # Window (17, 45] for inspections: day 30 only.
    assert feats["insp_count28"] == 1.0
    # Part swap at day 40 is inside both (17, 45] and (38, 45].
    assert feats["swap_any28"] == 1.0
    assert feats["swap_count7"] == 1.0
    # Scheduled service at day 33 is inside (17, 45] but not (38, 45].
    assert feats["pm_count28"] == 1.0
    assert feats["pm_count7"] == 0.0


def test_point_in_time_invariance():
    unit = _hand_unit()
    before = extract_features(unit, 20)
    trimmed = Unit(
        "hand",
        100,
        90,
        unit.sensors,
        [(d, k) for (d, k) in unit.events if d <= 20],
    )
    after = extract_features(trimmed, 20)
    assert before == after


def test_sensor_features_from_series():
    feats = extract_features(_hand_unit(), 20)
    assert feats["vib_level"] == 20.0
    assert feats["vib_slope7"] == 1.0
    assert feats["temp_level"] == 50.0


def test_build_cases_labels(fleet):
    cases = build_cases(fleet)
    lookup = dict((u.unit_id, u) for u in fleet)
    for case in cases:
        unit = lookup[case.unit_id]
        expected = 1 if unit.failure_time - case.day <= HORIZON else 0
        assert case.label == expected


def test_build_cases_grid(fleet):
    cases = build_cases(fleet)
    days = [c.day for c in cases if c.unit_id == fleet[0].unit_id]
    assert days[0] == 7
    assert all(b - a == 2 for a, b in zip(days, days[1:]))
    assert days[-1] < fleet[0].failure_time


def test_permuted_index_preserves_counts(fleet):
    rng = random.Random(0)
    for unit in fleet[:5]:
        real = event_index_of(unit)
        perm = permuted_event_index(unit, rng)
        for kind in real:
            assert len(perm[kind]) == len(real[kind])


def test_permuted_index_days_in_range(fleet):
    rng = random.Random(1)
    for unit in fleet[:5]:
        perm = permuted_event_index(unit, rng)
        for kind, days in perm.items():
            assert days == sorted(days)
            assert all(0 <= d < unit.failure_time for d in days)


def test_permuted_index_destroys_alignment(fleet):
    # After permutation, events should no longer concentrate in the
    # suspicion window: the share of events in the last `lead` days
    # should fall well below the real share.
    rng = random.Random(2)
    real_share = []
    perm_share = []
    for unit in fleet:
        if not unit.events:
            continue
        real_share.append(
            sum(1 for d, _k in unit.events if d >= unit.suspicion_time)
            / float(len(unit.events))
        )
        perm = permuted_event_index(unit, rng)
        total = sum(len(v) for v in perm.values())
        late = sum(
            1 for days in perm.values() for d in days if d >= unit.suspicion_time
        )
        perm_share.append(late / float(total))
    assert sum(real_share) / len(real_share) > 0.5
    assert sum(perm_share) / len(perm_share) < 0.25
