"""Core data types and shared constants for the leakage probe.

A Unit is one machine's complete run-to-failure record: daily sensor
series plus a maintenance-event log. The ground truth is always known
in this probe, because the fleet is synthetic: each unit carries a
hidden suspicion time, the moment the maintenance crew started acting
on an unrecorded suspicion of the coming failure. Events after that
time are intervention leakage by construction; sensor families are
genuine signal by construction. The probe never reads the suspicion
time - it has to recover the leak from the event timeline alone.
"""

import math

# Prediction task: at each case day, will the unit fail within HORIZON
# days? Features use only data up to and including the case day
# (point-in-time correct). The leak this probe hunts is therefore NOT
# a timestamp bug: the features are honestly computed, but the events
# they count were caused by the label.
HORIZON = 14
CASE_START_DAY = 7
CASE_STRIDE = 2
WINDOW_SHORT = 7
WINDOW_LONG = 28

EVENT_KINDS = (
    "work_order",
    "emergency_order",
    "inspection",
    "part_swap",
    "scheduled_service",
)

# Feature families. "source" says where the family's numbers come from:
# sensor families are read off the machine; log families are counted
# from the maintenance-event timeline - only log families can carry
# intervention leakage, and only they are touched by the permutation.
FAMILIES = {
    "vibration": {
        "source": "sensor",
        "leak": False,
        "features": ("vib_level", "vib_slope7"),
        "event_kinds": (),
    },
    "thermal": {
        "source": "sensor",
        "leak": False,
        "features": ("temp_level", "temp_slope7"),
        "event_kinds": (),
    },
    "load": {
        "source": "sensor",
        "leak": False,
        "features": ("load_level", "ambient_level"),
        "event_kinds": (),
    },
    "work_orders": {
        "source": "log",
        "leak": True,
        "features": ("wo_count7", "wo_emergency7"),
        "event_kinds": ("work_order", "emergency_order"),
    },
    "inspections": {
        "source": "log",
        "leak": True,
        "features": ("insp_count7", "insp_count28"),
        "event_kinds": ("inspection",),
    },
    "part_swaps": {
        "source": "log",
        "leak": True,
        "features": ("swap_count7", "swap_any28"),
        "event_kinds": ("part_swap",),
    },
    # The control family: a log family that does NOT leak. Scheduled
    # preventive maintenance runs on a fixed calendar, independent of
    # the suspicion signal, so its timestamps carry no alignment with
    # failure. The probe's precision is earned against this family -
    # a tool that flags every log family would flag it too.
    "scheduled_maintenance": {
        "source": "log",
        "leak": False,
        "features": ("pm_count7", "pm_count28"),
        "event_kinds": ("scheduled_service",),
    },
}

FAMILY_NAMES = tuple(FAMILIES.keys())
LEAK_FAMILIES = tuple(name for name in FAMILY_NAMES if FAMILIES[name]["leak"])
CLEAN_FAMILIES = tuple(name for name in FAMILY_NAMES if not FAMILIES[name]["leak"])
ALL_FEATURES = tuple(
    feature for name in FAMILY_NAMES for feature in FAMILIES[name]["features"]
)


def family_of_feature(feature):
    for name in FAMILY_NAMES:
        if feature in FAMILIES[name]["features"]:
            return name
    raise KeyError(feature)


class Unit(object):
    """One unit's complete run-to-failure record.

    failure_time and suspicion_time are integer days. suspicion_time is
    ground truth for scoring the probe only; feature extraction never
    sees it. events is a list of (day, kind) pairs sorted by day.
    sensors maps series name -> list of daily values, index = day.
    """

    def __init__(self, unit_id, failure_time, suspicion_time, sensors, events):
        self.unit_id = unit_id
        self.failure_time = failure_time
        self.suspicion_time = suspicion_time
        self.sensors = sensors
        self.events = sorted(events)

    @property
    def lead(self):
        """True suspicion lead time: days before failure the crew acted."""
        return self.failure_time - self.suspicion_time

    def event_days(self, kinds):
        return [day for (day, kind) in self.events if kind in kinds]

    def to_dict(self):
        return {
            "unit_id": self.unit_id,
            "failure_time": self.failure_time,
            "suspicion_time": self.suspicion_time,
            "lead": self.lead,
            "n_events": len(self.events),
        }


class Case(object):
    """One prediction case: a unit observed on one day.

    label is 1 when the unit fails within HORIZON days of the case day.
    features maps feature name -> float, computed point-in-time.
    """

    def __init__(self, unit_id, day, label, features):
        self.unit_id = unit_id
        self.day = day
        self.label = label
        self.features = features

    def key(self):
        return "%s@%d" % (self.unit_id, self.day)


def auc(scores, labels):
    """Rank-based AUC (Mann-Whitney), ties handled by mid-ranks."""
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    n = len(pairs)
    n_pos = sum(1 for _s, y in pairs if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        mid_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum_pos += mid_rank
        i = j + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mean(values):
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def sd(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) * (v - m) for v in values) / (len(values) - 1))
