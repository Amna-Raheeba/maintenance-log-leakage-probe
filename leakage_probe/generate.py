"""Seeded fleet simulator with a hidden suspicion signal.

Each unit degrades on a smooth health curve and fails at a known day.
The sensors see the degradation itself - genuine early-warning signal.
The maintenance log sees something else: a hidden suspicion time,
`lead` days before failure, from which the crew's behaviour changes.
Work orders, inspections and part swaps become far more frequent after
suspicion, because the crew already thinks the machine is dying. Any
model trained on log features is partly learning "the mechanic knew
first" - intervention leakage.

Routine maintenance also happens at a low background rate for the
whole life, so the log families are not trivially zero before
suspicion: the leak is the *rate step* aligned to the suspicion time,
and only a test that attacks the alignment can see it.
"""

import random

# Background (routine) daily event probabilities, active all life.
ROUTINE_RATES = {
    "work_order": 0.02,
    "emergency_order": 0.002,
    "inspection": 0.015,
    "part_swap": 0.005,
}

# Extra daily event probabilities once suspicion has switched on.
SUSPICION_RATES = {
    "work_order": 0.55,
    "emergency_order": 0.18,
    "inspection": 0.45,
    "part_swap": 0.16,
}

EVENT_KINDS = ("work_order", "emergency_order", "inspection", "part_swap")


def _generate_unit(rng, unit_id, mean_lead):
    failure_time = int(rng.uniform(80, 160))
    lead = max(1, int(round(rng.gauss(mean_lead, max(1.0, 0.25 * mean_lead)))))
    if lead >= failure_time:
        lead = failure_time // 4
    suspicion_time = failure_time - lead

    vib_offset = rng.uniform(-0.3, 0.3)
    vib_gain = rng.uniform(2.0, 2.8)
    temp_offset = rng.uniform(-1.5, 1.5)
    load_offset = rng.uniform(-5.0, 5.0)
    vib = []
    temp = []
    load = []
    ambient = []
    for t in range(failure_time):
        health = (t / float(failure_time)) ** 2.2
        vib.append(2.0 + vib_offset + vib_gain * health + rng.gauss(0.0, 1.1))
        temp.append(55.0 + temp_offset + 7.0 * health + rng.gauss(0.0, 4.0))
        load.append(60.0 + load_offset + rng.gauss(0.0, 8.0))
        ambient.append(22.0 + rng.gauss(0.0, 3.0))

    events = []
    for t in range(failure_time):
        suspected = t >= suspicion_time
        for kind in EVENT_KINDS:
            rate = ROUTINE_RATES[kind]
            if suspected:
                rate += SUSPICION_RATES[kind]
            if rng.random() < rate:
                events.append((t, kind))

    # Scheduled preventive maintenance: a fixed calendar (every ~30
    # days, jittered), drawn independently of degradation and of the
    # suspicion signal. This is the control family - log data whose
    # timestamps are honest.
    interval = rng.randint(26, 34)
    day = rng.randint(5, interval)
    while day < failure_time:
        events.append((day, "scheduled_service"))
        day += interval + rng.randint(-1, 1)

    from .models import Unit

    return Unit(
        unit_id,
        failure_time,
        suspicion_time,
        {"vib": vib, "temp": temp, "load": load, "ambient": ambient},
        events,
    )


def generate_fleet(seed, n_units=60, mean_lead=7):
    """A seeded fleet of run-to-failure units.

    mean_lead is the fleet-level suspicion lead time in days: on
    average, the crew starts acting `mean_lead` days before failure.
    Per-unit leads vary around it (Gaussian, sd 25%).
    """
    rng = random.Random(seed)
    units = []
    for i in range(n_units):
        units.append(_generate_unit(rng, "unit-%03d" % (i + 1), mean_lead))
    return units
