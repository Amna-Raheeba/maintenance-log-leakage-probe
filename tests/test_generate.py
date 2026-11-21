from leakage_probe.generate import ROUTINE_RATES, generate_fleet
from leakage_probe.models import mean


def test_deterministic_same_seed():
    a = generate_fleet(11, 10, 7)
    b = generate_fleet(11, 10, 7)
    for ua, ub in zip(a, b):
        assert ua.failure_time == ub.failure_time
        assert ua.suspicion_time == ub.suspicion_time
        assert ua.events == ub.events
        assert ua.sensors["vib"] == ub.sensors["vib"]


def test_different_seed_differs():
    a = generate_fleet(11, 10, 7)
    b = generate_fleet(12, 10, 7)
    assert [u.failure_time for u in a] != [u.failure_time for u in b]


def test_fleet_size_and_ids():
    fleet = generate_fleet(3, 25, 7)
    assert len(fleet) == 25
    assert fleet[0].unit_id == "unit-001"
    assert fleet[-1].unit_id == "unit-025"


def test_suspicion_is_failure_minus_lead():
    fleet = generate_fleet(5, 40, 7)
    for unit in fleet:
        assert unit.suspicion_time == unit.failure_time - unit.lead
        assert unit.lead >= 1


def test_mean_lead_tracks_scenario():
    for scenario in (3, 7, 14):
        fleet = generate_fleet(9, 60, scenario)
        assert abs(mean([u.lead for u in fleet]) - scenario) < 1.5


def test_sensor_series_lengths():
    fleet = generate_fleet(5, 10, 7)
    for unit in fleet:
        for series in ("vib", "temp", "load", "ambient"):
            assert len(unit.sensors[series]) == unit.failure_time


def test_events_sorted_and_in_range():
    fleet = generate_fleet(5, 10, 7)
    for unit in fleet:
        days = [d for d, _k in unit.events]
        assert days == sorted(days)
        assert all(0 <= d < unit.failure_time for d in days)


def test_event_rate_steps_up_after_suspicion():
    fleet = generate_fleet(5, 60, 7)
    pre = post = pre_days = post_days = 0
    for unit in fleet:
        pre_days += unit.suspicion_time
        post_days += unit.lead
        for day, _kind in unit.events:
            if day >= unit.suspicion_time:
                post += 1
            else:
                pre += 1
    assert post / float(post_days) > 5 * (pre / float(pre_days))


def test_routine_events_exist_before_suspicion():
    fleet = generate_fleet(5, 60, 7)
    pre = sum(
        1
        for unit in fleet
        for day, _k in unit.events
        if day < unit.suspicion_time
    )
    assert pre > 100


def test_vibration_rises_with_degradation():
    fleet = generate_fleet(5, 60, 7)
    early = []
    late = []
    for unit in fleet:
        early.extend(unit.sensors["vib"][:10])
        late.extend(unit.sensors["vib"][-10:])
    assert mean(late) > mean(early) + 1.0


def test_routine_rates_are_small():
    for rate in ROUTINE_RATES.values():
        assert rate < 0.05


def test_scheduled_services_roughly_periodic():
    fleet = generate_fleet(5, 10, 7)
    for unit in fleet:
        days = [d for d, k in unit.events if k == "scheduled_service"]
        assert len(days) >= 2
        gaps = [b - a for a, b in zip(days, days[1:])]
        assert all(24 <= g <= 37 for g in gaps)


def test_scheduled_services_ignore_suspicion():
    # Services fall both before and after the suspicion time - the
    # calendar does not know the machine is dying.
    fleet = generate_fleet(5, 40, 7)
    pre = sum(
        1
        for u in fleet
        for d, k in u.events
        if k == "scheduled_service" and d < u.suspicion_time
    )
    post = sum(
        1
        for u in fleet
        for d, k in u.events
        if k == "scheduled_service" and d >= u.suspicion_time
    )
    assert pre > 50
    assert post > 0
