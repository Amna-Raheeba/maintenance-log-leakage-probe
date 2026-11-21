# Changelog

## 0.1.0 - 2026-11-21

- Initial release: seeded run-to-failure fleet simulator with a
  hidden suspicion signal - maintenance behaviour (work orders,
  inspections, part swaps) steps up at a known lead time before
  failure, while sensor families carry genuine degradation signal.
- Event-permutation null: each event kind keeps its exact count per
  unit but its timestamps are re-drawn uniformly over the unit's
  life; a family whose AUC collapses under the null was carrying
  intervention leakage. Per-family leakage score, z-value, verdict.
- Lead-time leakage curve: pooled event rate by days before failure,
  with a half-rise estimator of the suspicion lead time.
- Collapse experiment: stdlib logistic regression scored with all
  features vs the probe-cleaned feature set on a held-out fleet.
- Baseline comparison: the naive "flag any family with AUC >= 0.70"
  screen, which flags strong sensors as leaks and misses weak leaks.
- Streamlit dashboard (3 tabs), CLI demo/evaluate commands, and a
  reproduction script with a headline-drift guard.
