"""Point-in-time feature extraction and the event permutation.

Features for a case on day t use only data from days <= t. That is
deliberate: the leak under study survives honest point-in-time joins,
because it lives in *why* the events happened, not in *when they were
recorded*.

The permutation is the probe's weapon. Within each unit, every
event kind keeps its exact count but its timestamps are re-drawn
uniformly over the unit's life: the alignment between the event
timeline and the failure time is destroyed, the event volume is not.
Sensor series are never touched, so a sensor family's predictive
power is identical under permutation, while a log family that was
riding the suspicion alignment collapses to its background-rate level.
"""

import bisect

from .models import (
    CASE_START_DAY,
    CASE_STRIDE,
    HORIZON,
    WINDOW_LONG,
    WINDOW_SHORT,
    Case,
)

KINDS = (
    "work_order",
    "emergency_order",
    "inspection",
    "part_swap",
    "scheduled_service",
)


def event_index_of(unit):
    """{kind: sorted event days} for a unit's real event log."""
    index = dict((kind, []) for kind in KINDS)
    for day, kind in unit.events:
        index[kind].append(day)
    return index


def permuted_event_index(unit, rng):
    """{kind: sorted days} with the unit's event timestamps shuffled.

    Each kind keeps its exact event count, but its days are re-drawn
    uniformly over the unit's life: the timestamps are permuted
    against the timeline. (Re-dealing the existing days among the
    events would NOT be a valid null - the multiset of event days is
    itself concentrated near failure, which is the leak.) Under this
    null a family keeps its overall event volume but loses all
    alignment with the failure time.
    """
    counts = dict((kind, 0) for kind in KINDS)
    for _day, kind in unit.events:
        counts[kind] += 1
    index = {}
    for kind in KINDS:
        n = counts[kind]
        if n == 0:
            index[kind] = []
        else:
            index[kind] = sorted(rng.sample(range(unit.failure_time), n))
    return index


def _count_between(sorted_days, lo, hi):
    """Count of days in (lo, hi]."""
    return bisect.bisect_right(sorted_days, hi) - bisect.bisect_right(sorted_days, lo)


def extract_features(unit, day, event_index=None):
    """All features for `unit` as observed on `day` (point-in-time).

    event_index optionally supplies an alternative {kind: sorted days}
    (e.g. a permuted timeline); sensor features are unaffected by it.
    """
    if event_index is None:
        event_index = event_index_of(unit)
    vib = unit.sensors["vib"]
    temp = unit.sensors["temp"]
    load = unit.sensors["load"]
    ambient = unit.sensors["ambient"]
    back = day - WINDOW_SHORT
    return {
        "vib_level": vib[day],
        "vib_slope7": (vib[day] - vib[back]) / float(WINDOW_SHORT),
        "temp_level": temp[day],
        "temp_slope7": (temp[day] - temp[back]) / float(WINDOW_SHORT),
        "load_level": load[day],
        "ambient_level": ambient[day],
        "wo_count7": float(
            _count_between(event_index["work_order"], day - WINDOW_SHORT, day)
        ),
        "wo_emergency7": float(
            _count_between(event_index["emergency_order"], day - WINDOW_SHORT, day)
        ),
        "insp_count7": float(
            _count_between(event_index["inspection"], day - WINDOW_SHORT, day)
        ),
        "insp_count28": float(
            _count_between(event_index["inspection"], day - WINDOW_LONG, day)
        ),
        "swap_count7": float(
            _count_between(event_index["part_swap"], day - WINDOW_SHORT, day)
        ),
        "swap_any28": 1.0
        if _count_between(event_index["part_swap"], day - WINDOW_LONG, day) > 0
        else 0.0,
        "pm_count7": float(
            _count_between(event_index["scheduled_service"], day - WINDOW_SHORT, day)
        ),
        "pm_count28": float(
            _count_between(event_index["scheduled_service"], day - WINDOW_LONG, day)
        ),
    }


def build_cases(fleet):
    """One case per (unit, grid day) with point-in-time features."""
    cases = []
    for unit in fleet:
        day = CASE_START_DAY
        while day < unit.failure_time:
            label = 1 if unit.failure_time - day <= HORIZON else 0
            cases.append(Case(unit.unit_id, day, label, extract_features(unit, day)))
            day += CASE_STRIDE
    return cases


def unit_lookup(fleet):
    return dict((unit.unit_id, unit) for unit in fleet)
