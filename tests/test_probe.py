from leakage_probe.generate import generate_fleet
from leakage_probe.models import CLEAN_FAMILIES, LEAK_FAMILIES, mean
from leakage_probe.probe import baseline_screen, estimate_lead, leakage_curve


def test_probe_flags_all_leak_families(probe_results):
    for family in LEAK_FAMILIES:
        assert probe_results[family]["flagged"], family


def test_probe_clears_all_clean_families(probe_results):
    for family in CLEAN_FAMILIES:
        assert not probe_results[family]["flagged"], family


def test_leakage_scores_positive_for_log_families(probe_results):
    for family in LEAK_FAMILIES:
        assert probe_results[family]["leakage_score"] > 0.08, family
        assert probe_results[family]["z"] > 3.0, family


def test_sensor_permutation_is_identity(probe_results):
    # The permutation never touches sensor series, so a sensor
    # family's permuted AUC equals its real AUC exactly.
    for family in ("vibration", "thermal", "load"):
        r = probe_results[family]
        assert r["perm_mean_auc"] == r["real_auc"]
        assert r["leakage_score"] == 0.0


def test_control_log_family_not_flagged(probe_results):
    # Scheduled maintenance is a log family with honest timestamps:
    # permuting them changes nothing, so its leakage score stays
    # near zero and it must not be flagged.
    r = probe_results["scheduled_maintenance"]
    assert not r["flagged"]
    assert r["leakage_score"] < 0.05


def test_permuted_auc_near_chance_for_leak_families(probe_results):
    for family in LEAK_FAMILIES:
        assert probe_results[family]["perm_mean_auc"] < 0.58, family


def test_real_auc_strong_for_vibration(probe_results):
    assert probe_results["vibration"]["real_auc"] > 0.75


def test_load_family_is_chance(probe_results):
    assert abs(probe_results["load"]["real_auc"] - 0.5) < 0.06


def test_baseline_screen_flags_a_clean_family(cases):
    # The naive screen's known failure mode: it cannot tell a strong
    # sensor from a leak, so it flags at least one clean family.
    flagged = baseline_screen(cases)
    assert any(family in CLEAN_FAMILIES for family in flagged)


def test_baseline_screen_misses_weakest_leak(cases):
    flagged = baseline_screen(cases)
    assert "part_swaps" not in flagged


def test_estimate_lead_close_for_all_log_families():
    fleet = generate_fleet(5, 60, 7)
    true_mean = mean([u.lead for u in fleet])
    for family in LEAK_FAMILIES:
        est, _rates, _base = estimate_lead(fleet, family)
        assert abs(est - true_mean) <= 3.0, (family, est, true_mean)


def test_estimate_lead_tracks_scenario():
    for scenario in (3, 14):
        fleet = generate_fleet(9, 60, scenario)
        true_mean = mean([u.lead for u in fleet])
        est, _r, _b = estimate_lead(fleet, "work_orders")
        assert abs(est - true_mean) <= 3.5, (scenario, est, true_mean)


def test_leakage_curve_step_shape():
    fleet = generate_fleet(5, 60, 7)
    rates, baseline = leakage_curve(fleet, "work_orders")
    assert rates[1] > 3 * baseline
    assert rates[25] < rates[1]


def test_leakage_curve_sensor_family_is_zero():
    fleet = generate_fleet(5, 10, 7)
    rates, baseline = leakage_curve(fleet, "vibration")
    assert all(r == 0.0 for r in rates)
    assert baseline == 0.0


def test_probe_fleet_covers_all_families(probe_results):
    assert len(probe_results) == 7
    for result in probe_results.values():
        assert 0.0 <= result["real_auc"] <= 1.0
