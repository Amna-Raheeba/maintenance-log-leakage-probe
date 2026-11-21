# Maintenance-Log Leakage Probe

## 1. What this is

A probe for the predictive-maintenance leak that point-in-time joins
cannot catch: **the maintenance actions themselves leak the failure
label**. When a crew starts to suspect a machine is dying, they file
more work orders, inspect it more often, and swap parts early. Those
actions are recorded in the maintenance log, the log becomes features,
and the model learns "the mechanic knew first" — performance that
evaporates the moment the model is deployed somewhere the crew has
not already done its job.

Every feature in this project is computed point-in-time correctly.
The leak survives anyway, because it lives in *why* the events
happened, not in *when they were recorded*. The probe detects it with
an **event-permutation null**: shuffle the maintenance-event
timestamps within each unit, and a feature family whose predictive
power collapses was carrying intervention leakage. A family whose
power survives the shuffle is reading the machine, not the crew.

## 2. The novel mechanism

1. **Align every feature family to the maintenance-event timeline.**
   Families are declared by provenance: sensor families (vibration,
   thermal, load) are read off the machine; log families (work
   orders, inspections, part swaps) are counted from the event log.
2. **Run the event-permutation null per family.** Each event kind
   keeps its exact count per unit, but its timestamps are re-drawn
   uniformly over the unit's life — same event volume, alignment
   destroyed. The family's scoring rule (its strongest feature, or
   the standardised aggregate) is chosen once on the real alignment
   and held fixed across 20 permutations. The **leakage score** is
   real AUC minus mean permuted AUC, with a z-value from the
   permutation spread. A family is flagged at score >= 0.05, real
   AUC >= 0.58, z >= 2.5.
   (Re-dealing the existing event days among the events would *not*
   be a valid null: the multiset of event days is itself concentrated
   near failure, and that concentration is the leak.)
3. **Read off the lead-time leakage curve.** Pool the family's event
   rate by days before failure. For a leaking family the curve is a
   step — background rate far out, elevated rate inside the suspicion
   window — and the step's half-rise crossing estimates *how many
   days before failure the leak switches on*, separating legitimate
   early-warning signal from post-suspicion contamination.
4. **Price the leak.** Train one model on all features and one on
   the probe-cleaned feature set, score both on a held-out fleet:
   the AUC difference (the **collapse**) is the part of the reported
   benchmark score that was the crew's suspicion, memorised.

## 3. Prior art and the differentiator

- **Kapoor & Narayanan's leakage taxonomy (2023)** names the leakage
  families and is the standard citation — but it is a taxonomy and a
  set of case studies, not a test. It cannot look at *your* dataset
  and say which family leaks or when the leak switches on.
- **Generic RUL leakage-audit advice** (point-in-time joins,
  window-vs-cutoff checks, train/test separation by unit) catches
  *timestamp* leakage in feature windows. The intervention leak is
  not a timestamp bug: every join here is honest, and the features
  still leak.
- **Single-feature AUC screening** ("flag anything suspiciously
  predictive") is the common practical fallback. It cannot tell a
  strong sensor from a leak — both are just predictive — and the
  evaluation below shows it flagging clean sensor families while
  missing the weakest real leak.

**Differentiator:** no existing tool tests intervention-event
alignment with a permutation null, and none produces a lead-time
leakage curve for maintenance logs. This probe does both, per feature
family, against planted ground truth.

## 4. The fleet simulator

- Seeded synthetic fleet (`leakage_probe/generate.py`): each unit
  degrades on a smooth health curve `h(t) = (t/T)^2.2`, fails at a
  known day `T` (80–160 days), and is observed daily.
- A hidden **suspicion time** sits `lead` days before failure. From
  that day the crew's event rates step up sharply (e.g. work orders
  0.02 → 0.57 per day). The suspicion time is ground truth for
  scoring only; feature extraction never sees it.
- Routine maintenance runs at a low background rate all life, so log
  families are not trivially zero before suspicion — the leak is the
  *rate step*, and only a test that attacks the alignment can see it.
- **Scheduled preventive maintenance** runs on a fixed ~30-day
  calendar, independent of degradation and suspicion: a log family
  that does *not* leak, included as the control the probe's precision
  is earned against.
- Prediction task: at each case day (every 2 days from day 7), will
  the unit fail within 14 days? Features use trailing 7/28-day
  windows ending on the case day.
- Scenarios are paired: the same seed produces identical machines
  and sensor streams at every suspicion lead — only the suspicion
  time moves — so lead-scenario comparisons are within-fleet.

## 5. Feature families

| family | source | truth | features |
|---|---|---|---|
| vibration | sensor | clean | vib level, 7-day slope |
| thermal | sensor | clean | temperature level, 7-day slope |
| load | sensor | clean (null) | load level, ambient level |
| work_orders | log | **leak** | work orders in 7d, emergency orders in 7d |
| inspections | log | **leak** | inspections in 7d, in 28d |
| part_swaps | log | **leak** | swaps in 7d, any swap in 28d |
| scheduled_maintenance | log | clean (control) | services in 7d, in 28d |

## 6. Headline results (measured)

15 seeded runs (5 seeds × suspicion leads 3 / 7 / 14 days), 60 units
per fleet, ~3,300 cases per fleet. Full tables in `results/`.

**Detection — probe vs the naive AUC screen:**

| method | precision | recall |
|---|---|---|
| event-permutation probe | **1.000** | **0.733** |
| baseline: flag any family with AUC >= 0.70 | 0.423 | 0.489 |

The baseline flags the two strong sensor families as leaks in most
runs and never catches part swaps (mean AUC 0.657, below its
threshold). The probe never flags a clean family in 15 runs —
including the scheduled-maintenance control (mean leakage score
0.026, flagged 0/15) — and flags work orders in 12/15 runs,
inspections 11/15, part swaps 10/15.

**The recall limit is graded and honest:** every probe miss is at
the 3-day suspicion lead, where the leak is active for only 3 of the
14 horizon days and a family's pooled AUC sits at 0.53–0.58 — too
faint to pass a materiality threshold. At leads 7 and 14, detection
is perfect (30/30 family-runs).

**Lead-time estimation** (half-rise of the leakage curve vs the true
fleet mean lead): MAE 0.72 days at lead 3, 0.88 at lead 7, 1.41 at
lead 14 — **1.00 day overall**, with no access to the suspicion
times.

**Collapse** (held-out AUC, all features → probe-cleaned):

| suspicion lead | AUC all | AUC cleaned | collapse | AUC log only |
|---|---|---|---|---|
| 3 days | 0.909 | 0.908 | 0.001 | 0.578 |
| 7 days | 0.928 | 0.905 | 0.023 | 0.737 |
| 14 days | 0.976 | 0.905 | 0.071 | 0.935 |

Mean collapse **0.031 AUC** (0.938 → 0.906). When suspicion covers
the whole horizon (lead 14), the maintenance log *alone* (0.935)
nearly matches the full model and beats every sensor combined
(0.905) — the benchmark is then mostly measuring the crew.

Demo fleet (seed 7, lead 7): work orders AUC 0.711 → permuted 0.514
(score 0.197, z 6.6); inspections 0.722 → 0.520 (z 5.2); part swaps
0.597 → 0.499 (z 7.3); scheduled maintenance 0.554 → 0.514
(score 0.040, z 0.9, clean). Lead estimates 8 / 7 / 8 days vs a true
mean of 7.0.

## 7. Honest limits

- **Synthetic fleet.** Suspicion is a step function at one lead time
  per unit; real suspicion is gradual, partial, and sometimes wrong
  (crews also act on machines that recover). The probe validates
  against planted ground truth because real logs do not record when
  suspicion started — that absence is the problem the tool addresses,
  but the measured numbers are simulator numbers.
- **Recall is limited by coverage.** A leak active for a small share
  of the prediction horizon moves pooled AUC very little; the probe
  misses most 3-day leaks at a 14-day horizon (Section 6). Shortening
  the horizon or scoring per-lead-bucket would raise sensitivity and
  is the obvious next step.
- **Provenance is required.** The probe must know which families are
  counted from the event log — metadata any real dataset has — but
  it cannot audit a feature of unknown provenance.
- **The null assumes within-unit exchangeability.** Calendar effects
  shared by the event process and the failure process (e.g. weekday
  shutdowns) could survive the shuffle; the scheduled-maintenance
  control covers fixed calendars, not failure-correlated ones.
- **The collapse is model-dependent.** It is measured with a small
  stdlib logistic regression; a stronger model would extract more
  from both the sensors and the leak, changing the split, though not
  the detection result.

## 8. How to run

```bash
pip install pytest streamlit   # streamlit only for the dashboard
python -m leakage_probe demo                    # one fleet, full probe report
python -m leakage_probe evaluate --out results  # the headline evaluation
python scripts/reproduce_results.py             # re-run + drift guard
python notebooks/01_demo_run.py
python notebooks/02_evaluation_experiment.py
streamlit run app.py                            # dashboard (3 tabs)
pytest                                          # 68 tests
```

## 9. Repository layout

- `leakage_probe/` — `models` (families, cases, AUC), `generate`
  (fleet simulator with hidden suspicion), `features` (point-in-time
  extraction + the permutation), `probe` (permutation test, leakage
  curve, lead estimator, baseline screen), `classifier` (stdlib
  logistic regression + collapse experiment), `evaluate`, `report`,
  `cli`.
- `data/cases.json` — demo-fleet summary + family ground truth.
- `examples/` — runnable demo, one unit's full timeline, sample
  report. `notebooks/` — demo and evaluation as runnable scripts.
- `results/` — the measured headline evaluation (JSON / CSV /
  summary / headline). `scripts/` — data builder + reproduction
  guard. `tests/` — 68 tests.

*Author: Amna Raheeba. MIT license.*
