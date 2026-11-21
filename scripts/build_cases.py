"""Build the seeded demo fleet summary stored in data/cases.json.

The stored file is a compact summary (one row per unit) of the demo
fleet (seed 7, suspicion lead scenario 7 days) plus the family
ground-truth table. Full trajectories are always regenerated
deterministically from the seed, never stored.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leakage_probe.generate import generate_fleet
from leakage_probe.models import FAMILIES, FAMILY_NAMES


def main():
    fleet = generate_fleet(7, 120, 7)
    payload = {
        "seed": 7,
        "mean_lead_scenario": 7,
        "n_units": len(fleet),
        "families": dict(
            (
                name,
                {
                    "source": FAMILIES[name]["source"],
                    "leak_ground_truth": FAMILIES[name]["leak"],
                    "features": list(FAMILIES[name]["features"]),
                },
            )
            for name in FAMILY_NAMES
        ),
        "units": [unit.to_dict() for unit in fleet],
    }
    out = os.path.join(os.path.dirname(__file__), "..", "data", "cases.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print("Wrote %s (%d units)" % (out, len(fleet)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
