# Sample report

Output of `python -m leakage_probe demo` (seed 7, suspicion lead scenario 7 days):

```
Maintenance-log leakage probe - seed 7, suspicion lead scenario 7 days (true fleet mean lead 7.0 days)
Fleet: 60 units, 3261 prediction cases (fail within 14 days = 1)

family       truth      AUC permAUC    leak      z  verdict
vibration    clean    0.823   0.823   0.000    0.0  clean
thermal      clean    0.749   0.749   0.000    0.0  clean
load         clean    0.510   0.510   0.000    0.0  clean
work_orders  leak     0.711   0.514   0.197    6.6  LEAK
inspections  leak     0.722   0.520   0.202    5.2  LEAK
part_swaps   leak     0.597   0.499   0.098    7.3  LEAK
scheduled_maintenance clean    0.554   0.514   0.040    0.9  clean

Naive AUC screen flagged: inspections, thermal, vibration, work_orders
Probe flagged: work_orders, inspections, part_swaps

Lead-time estimates (days before failure the leak switches on):
  work_orders  estimated 8.0  (true mean 7.0)
  inspections  estimated 7.0  (true mean 7.0)
  part_swaps   estimated 8.0  (true mean 7.0)

Held-out AUC: all features 0.923 | probe-cleaned 0.909 | sensors only 0.909 | log only 0.718
Collapse when the flagged families are removed: 0.014 AUC
```
