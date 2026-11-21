# Evaluation summary

Seeded fleets: 15 runs (5 seeds x suspicion leads 3, 7, 14), 60 units per fleet. Ground truth: three maintenance-log families leak (work orders, inspections, part swaps); the three sensor families and the scheduled-maintenance control family are clean.

## Detection: probe vs naive AUC screen

- Probe (event-permutation null): precision 1.000, recall 0.733
- Baseline (flag any family with AUC >= 0.70): precision 0.423, recall 0.489

## Per family (pooled over all runs)

| family | truth | mean AUC | mean leakage score | flagged (runs) |
|---|---|---|---|---|
| vibration | clean | 0.811 | 0.000 | 0/15 |
| thermal | clean | 0.756 | 0.000 | 0/15 |
| load | clean | 0.510 | 0.000 | 0/15 |
| work_orders | leak | 0.735 | 0.231 | 12/15 |
| inspections | leak | 0.730 | 0.206 | 11/15 |
| part_swaps | leak | 0.657 | 0.145 | 10/15 |
| scheduled_maintenance | clean | 0.549 | 0.026 | 0/15 |

## Lead-time estimation and collapse, per suspicion lead

| true lead (days) | lead MAE | AUC all | AUC cleaned | collapse | AUC log only | AUC sensors only |
|---|---|---|---|---|---|---|
| 3 | 0.72 | 0.909 | 0.908 | 0.001 | 0.578 | 0.905 |
| 7 | 0.88 | 0.928 | 0.905 | 0.023 | 0.737 | 0.905 |
| 14 | 1.41 | 0.976 | 0.905 | 0.071 | 0.935 | 0.905 |

Mean lead-time estimation error: 1.00 days. Mean collapse: 0.031 AUC (all-features 0.938 -> cleaned 0.906).
