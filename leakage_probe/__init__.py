"""Maintenance-Log Leakage Probe.

Detects intervention leakage in predictive-maintenance datasets: the
leak where maintenance actions themselves (work orders, inspections,
part swaps) encode the failure label, because the crew acted on a
suspicion that is not itself a recorded feature. The probe aligns every
feature family to the maintenance-event timeline and runs an
event-permutation null: shuffle the event timestamps within each unit,
and a family whose predictive power collapses was carrying intervention
leakage rather than genuine early-warning signal.
"""

__version__ = "0.1.0"
__author__ = "Amna Raheeba"
