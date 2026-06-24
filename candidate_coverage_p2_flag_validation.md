# Candidate Coverage P2 Flag Validation

Date: 2026-06-24

Scope: JSON-only validation of the default-off P2 normalized drift admission flag in the candidate coverage pipeline. No model decoding, candidate generation, fingerprint simulation, reranking, formal eval, 64-sample eval, grids, retraining, scorer changes, rerank-mode changes, checkpoint/data changes, target-format changes, fingerprint changes, or production default changes were run.

## Summary

- Default-off flag present: `True`.
- Default behavior intended unchanged: `True`.
- Default source: `beam`.
- Uses reusable `scripts/normalized_drift_admission.py` hook: `True`.
- Independent canonicalizer duplication avoided: `True`.
- JSON reports validated: `4`.
- Metric cross-check status: `ok`.
- Schema status: `ok`.
- Canonicalizer safety status: `ok`.
- Mismatch count: `0`.
- Model-backed candidate generation avoided: `True`.
- Decision: `A` - Pipeline flag is ready for a future small local coverage-only run.

## Metric Cross-Checks

| Metric | Expected | Status | Source fields |
| --- | --- | --- | --- |
| `current_expanded_full_oracle` | `15` | ok | `pipeline_flag_block:current_expanded_full_oracle=15`<br>`candidate_coverage_p2_normalized_admission:summary.current_expanded_full_oracle=15` |
| `canonicalized_expanded_pool_coverage` | `23` | ok | `pipeline_flag_block:canonicalized_expanded_pool_coverage=23`<br>`candidate_coverage_p2_normalized_admission:summary.canonicalized_expanded_pool_coverage=23` |
| `p2_normalized_admission_coverage` | `30` | ok | `pipeline_flag_block:p2_normalized_admission_coverage=30`<br>`candidate_coverage_p2_normalized_admission:summary.p2_normalized_admission_coverage=30` |
| `p2_pair_count` | `340` | ok | `pipeline_flag_block:p2_pair_count=340`<br>`candidate_coverage_p2_normalized_admission:summary.p2_pair_count=340` |
| `newly_recovered` | `[0, 2, 6, 7, 15, 25, 29]` | ok | `pipeline_flag_block:newly_recovered=[0, 2, 6, 7, 15, 25, 29]`<br>`candidate_coverage_p2_normalized_admission:summary.newly_recovered_beyond_current=[0, 2, 6, 7, 15, 25, 29]` |
| `remaining_missing` | `[16, 28]` | ok | `pipeline_flag_block:remaining_missing=[16, 28]`<br>`candidate_coverage_p2_normalized_admission:summary.remaining_missing_samples=[16, 28]` |
| `sampling_excluded` | `True` | ok | `pipeline_flag_block:sampling_excluded=True`<br>`candidate_coverage_p2_normalized_admission:summary.sampling_excluded=True` |
| `sampling_recovered_canonical_drift_count` | `0` | ok | `pipeline_flag_block:sampling_recovered_canonical_drift_count=0`<br>`candidate_coverage_p2_normalized_admission:summary.sampling_recovered_canonical_drift_count=0` |
| `unsafe_collision_flag` | `False` | ok | `pipeline_flag_block:unsafe_collision_flag=False`<br>`candidate_coverage_p2_normalized_admission:safety.unsafe_collision_flag=False` |
| `metric_cross_check_status` | `ok` | ok | `hook_validation:summary.metric_cross_check_status=ok` |
| `canonicalizer_safety_status` | `ok` | ok | `hook_validation:summary.canonicalizer_safety_status=ok` |

## Schema Status

- Schema status: `ok`.
- Missing fields: `[]`.
- Field aliases used: `['pipeline_block.* from build_normalized_drift_admission_block', 'p2_json.summary.*', 'hook_validation.summary.*']`.

## Mismatches

- None.

## Recommendation

Run a future small local coverage-only candidate coverage command with the explicit P2 flag. No formal eval.

No formal eval is recommended.
