# P2 Normalized Admission Hook Validation

Date: 2026-06-23

Scope: offline JSON-only cross-validation of the reusable P2 normalized drift admission hook against prior helper diagnostics. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production default changes were run.

## Summary

- JSON reports validated: `6`.
- Metric cross-check status: `ok`.
- Schema status: `ok`.
- Canonicalizer safety status: `ok`.
- Mismatch count: `0`.
- Decision: `A` - Hook is internally consistent.

## Metric Cross-Checks

| Metric | Expected | Status | Source fields |
| --- | --- | --- | --- |
| `current_expanded_full_oracle` | `15` | ok | `p2:summary.current_expanded_full_oracle=15`<br>`capped:metadata.baseline.current_expanded_full_oracle_coverage=15`<br>`normalized_admission:summary.current_expanded_full_oracle_coverage=15`<br>`normalization:summary.current_expanded_full_oracle_coverage=15`<br>`expanded_coverage:derived sum(samples[].has_full)=15`<br>`recomputed_hook:derived=15` |
| `canonicalized_expanded_pool_coverage` | `23` | ok | `p2:summary.canonicalized_expanded_pool_coverage=23`<br>`capped:metadata.baseline.canonicalized_expanded_pool_coverage=23`<br>`normalized_admission:summary.canonicalized_expanded_pool_coverage=23`<br>`normalization:summary.canonicalized_expanded_pool_full_oracle_coverage=23`<br>`recomputed_hook:derived=23` |
| `full_normalized_admission_coverage` | `30` | ok | `p2:summary.full_normalized_admission_coverage=30`<br>`capped:metadata.baseline.full_normalized_drift_only_admission_coverage=30`<br>`normalized_admission:summary.normalized_drift_only_admission_coverage=30`<br>`normalization:summary.canonicalized_drift_only_admission_full_oracle_coverage=30` |
| `p2_normalized_admission_coverage` | `30` | ok | `p2:summary.p2_normalized_admission_coverage=30`<br>`capped:policies.P2_one_per_family.projected_full_oracle_coverage=30`<br>`recomputed_hook:derived=30` |
| `p2_pair_count` | `340` | ok | `p2:summary.p2_pair_count=340`<br>`capped:policies.P2_one_per_family.pair_count_after=340`<br>`recomputed_hook:derived=340` |
| `full_normalized_admission_pair_count` | `1105` | ok | `p2:summary.full_normalized_admission_pair_count=1105`<br>`capped:metadata.baseline.full_normalized_admission_target_pair_count=1105`<br>`normalized_admission:pair_pressure.pair_count_after_canonical_sum=1105` |
| `truth_aware_upper_bound_pair_count` | `320` | ok | `p2:summary.truth_aware_upper_bound_pair_count=320`<br>`capped:truth_aware_upper_bound.pair_count_after=320`<br>`capped:policies.P4_recovery_targeted_oracle_upper_bound.pair_count_after=320` |
| `newly_recovered` | `[0, 2, 6, 7, 15, 25, 29]` | ok | `p2:summary.newly_recovered_beyond_current=[0, 2, 6, 7, 15, 25, 29]`<br>`capped:policies.P2_one_per_family.newly_recovered_beyond_current_expanded_canonical_pool=[0, 2, 6, 7, 15, 25, 29]`<br>`normalized_admission:summary.newly_recovered_beyond_current_expanded_canonical_pool=[0, 2, 6, 7, 15, 25, 29]`<br>`recomputed_hook:derived=[0, 2, 6, 7, 15, 25, 29]` |
| `remaining_missing` | `[16, 28]` | ok | `p2:summary.remaining_missing_samples=[16, 28]`<br>`capped:policies.P2_one_per_family.remaining_missing_samples=[16, 28]`<br>`normalized_admission:summary.remaining_missing_samples=[16, 28]`<br>`recomputed_hook:derived=[16, 28]` |
| `p2_source_breakdown` | `{'beam': 15}` | ok | `p2:source_breakdown={'beam': 15}`<br>`capped:policies.P2_one_per_family.source_breakdown={'beam': 15}`<br>`normalized_admission:source_breakdown.drift_only_tail_normalized={'beam': 15}`<br>`recomputed_hook:derived={'beam': 15}` |
| `sampling_recovered_canonical_drift_count` | `0` | ok | `p2:summary.sampling_recovered_canonical_drift_count=0` |
| `sampling_can_be_dropped` | `True` | ok | `p2:summary.sampling_excluded=True`<br>`capped:summary.sampling_can_be_dropped_for_all_policies=True`<br>`capped:policies.P2_one_per_family.sampling_can_be_dropped=True`<br>`normalized_admission:not summary.sampling_contributes_recovered_canonical_drift=True` |
| `unsafe_collision_flag` | `False` | ok | `p2:safety.unsafe_collision_flag=False`<br>`capped:collision_safety.unsafe_collision_flag=False`<br>`normalized_admission:collision_safety.unsafe_collision_flag=False`<br>`normalization:collision_risk.unsafe=False` |

## Schema Status

| Report | Status | Missing fields | Aliases / derived fields |
| --- | --- | --- | --- |
| `candidate_coverage_p2_normalized_admission.json` | ok | - | `total sample count <- summary.total_samples_analyzed`<br>`target oracle-absent sample indices <- samples`<br>`current expanded oracle coverage <- summary.current_expanded_full_oracle`<br>`canonicalized expanded coverage <- summary.canonicalized_expanded_pool_coverage`<br>`full normalized admission coverage <- summary.full_normalized_admission_coverage`<br>`P2 coverage <- summary.p2_normalized_admission_coverage`<br>`P2 admitted candidates by sample <- admitted_candidates_by_sample`<br>`P2 family/source/rank metadata <- admitted_candidates_by_sample`<br>`exact diffusion presence <- samples`<br>`pair-count estimates <- summary.p2_pair_count`<br>`safety self-checks <- safety.self_checks` |
| `capped_normalized_drift_admission_diagnostics.json` | partial | P2 admitted candidates by sample | `total sample count <- summary.total_samples_analyzed`<br>`target oracle-absent sample indices <- samples`<br>`current expanded oracle coverage <- metadata.baseline.current_expanded_full_oracle_coverage`<br>`canonicalized expanded coverage <- metadata.baseline.canonicalized_expanded_pool_coverage`<br>`full normalized admission coverage <- metadata.baseline.full_normalized_drift_only_admission_coverage`<br>`P2 coverage <- policies.P2_one_per_family.projected_full_oracle_coverage`<br>`P2 family/source/rank metadata <- policies.P2_one_per_family.source_breakdown`<br>`exact diffusion presence <- samples`<br>`pair-count estimates <- policies.P2_one_per_family.pair_count_after`<br>`safety self-checks <- collision_safety.unsafe_checks` |
| `normalized_drift_only_admission_coverage_diagnostics.json` | partial | P2 coverage, P2 admitted candidates by sample, P2 family/source/rank metadata | `total sample count <- summary.total_samples_analyzed`<br>`target oracle-absent sample indices <- samples`<br>`current expanded oracle coverage <- summary.current_expanded_full_oracle_coverage`<br>`canonicalized expanded coverage <- summary.canonicalized_expanded_pool_coverage`<br>`full normalized admission coverage <- summary.normalized_drift_only_admission_coverage`<br>`exact diffusion presence <- samples`<br>`pair-count estimates <- pair_pressure.pair_count_after_canonical_sum`<br>`safety self-checks <- collision_safety.unsafe_checks` |
| `candidate_normalization_constant_folding_diagnostics.json` | partial | P2 coverage, P2 admitted candidates by sample, P2 family/source/rank metadata, pair-count estimates | `total sample count <- summary.total_samples_analyzed`<br>`target oracle-absent sample indices <- metadata.target_indices`<br>`current expanded oracle coverage <- summary.current_expanded_full_oracle_coverage`<br>`canonicalized expanded coverage <- summary.canonicalized_expanded_pool_full_oracle_coverage`<br>`full normalized admission coverage <- summary.canonicalized_drift_only_admission_full_oracle_coverage`<br>`exact diffusion presence <- samples`<br>`safety self-checks <- collision_risk.unsafe_checks` |
| `drift_only_candidate_diversity_smoke.json` | partial | total sample count, current expanded oracle coverage, canonicalized expanded coverage, full normalized admission coverage, P2 coverage, P2 admitted candidates by sample, pair-count estimates, safety self-checks | `target oracle-absent sample indices <- metadata.target_indices`<br>`P2 family/source/rank metadata <- samples`<br>`exact diffusion presence <- samples` |
| `candidate_coverage_pair_expanded.json` | partial | canonicalized expanded coverage, full normalized admission coverage, P2 coverage, P2 admitted candidates by sample, P2 family/source/rank metadata, safety self-checks | `total sample count <- samples`<br>`target oracle-absent sample indices <- samples where has_full is false (derived)`<br>`current expanded oracle coverage <- sum(samples[].has_full) (derived)`<br>`exact diffusion presence <- samples`<br>`pair-count estimates <- samples` |

## Canonicalizer Safety

- `pow2_x0_not_x0`: `True`
- `const_pow2_not_const_x0`: `True`
- `sin_x0_not_x0`: `True`
- `const_chain_linear_matches`: `True`
- `const_chain_sin_matches`: `True`

## Mismatches

- None.

## Recommendation

Add a future coverage-only candidate-coverage pipeline flag. No formal eval.

No formal eval is recommended.
