# Expanded Pairing 32-Sample Eval Report

Date: 2026-06-22

Log source:

```text
/private/tmp/gensr_sde_32_expanded_pairing_rolewise.log
```

Command status: completed. This was one formal 32-sample eval-only run from the
restored 2000-step checkpoint. No 64-sample eval, retraining, active/weak grid,
or scorer/default change was run.

## Main Metrics

| Metric | Baseline role-wise pairing | Expanded coverage-only | Expanded formal eval |
| --- | ---: | ---: | ---: |
| Selected exact/relaxed | 5/32 | n/a | 3/32 |
| Oracle exact/relaxed | 9/32 | 15/32 | 15/32 |
| Pair oracle exact/relaxed | 2/32 | 8/32 | 8/32 |

Expanded pairing successfully reproduced the coverage-only oracle ceiling in
formal eval: oracle exact/relaxed rose from `9/32` to `15/32`, and pair oracle
exact/relaxed rose from `2/32` to `8/32`.

However, selected reranked exact/relaxed dropped from `5/32` to `3/32`, so
expanded pairing should not become the default decoding setting yet.

## Detailed Metrics

| Metric | Value |
| --- | ---: |
| greedy_sequence_exact | 0.000000 |
| greedy_sequence_relaxed_no_constants | 0.000000 |
| constrained_beam_sequence_exact | 0.000000 |
| constrained_beam_sequence_relaxed_no_constants | 0.000000 |
| reranked_sequence_exact | 0.093750 |
| reranked_sequence_relaxed_no_constants | 0.093750 |
| rerank_oracle_sequence_exact | 0.468750 |
| rerank_oracle_sequence_relaxed_no_constants | 0.468750 |
| rerank_pair_oracle_sequence_exact | 0.250000 |
| rerank_pair_oracle_sequence_relaxed_no_constants | 0.250000 |
| rerank_candidates_attempted | 715 |
| rerank_candidates_valid | 715 |
| rerank_parse_failures | 0 |
| rerank_fingerprint_failures | 0 |
| rerank_unique_candidate_avg | 22.343750 |
| rerank_unique_drift_avg | 4.468750 |
| rerank_unique_diffusion_avg | 5.000000 |
| rerank_unique_paired_candidate_avg | 12.906250 |

Oracle source breakdown:

| Source | Best oracle count |
| --- | ---: |
| beam | 6 |
| sampling | 1 |
| pair | 8 |

Selected hit source breakdown:

| Source | Selected hit count |
| --- | ---: |
| beam | 2 |
| sampling | 0 |
| pair | 1 |

## Miss Diagnosis Summary

| Diagnostic | Count |
| --- | ---: |
| Samples with any oracle | 15 |
| Samples where selected is oracle | 3 |
| Samples where oracle exists but selected misses | 12 |
| Oracle-only-pair misses | 7 |
| Beam/sampling oracle score misses | 5 |
| Role-wise constant-grid rescues among listed misses | 0 |

The expanded candidate pool adds the intended paired oracle candidates, but the
current role-wise active+weak constant-grid scorer often ranks non-oracles above
those newly introduced paired oracles.

## Interpretation

Expanded pairing is diagnostically useful because it confirms the oracle ceiling
can be raised by recombining existing drift/diffusion spans. It is not yet a
selected-recovery improvement. The scorer/ranking stage is now the bottleneck:
formal oracle exact/relaxed is `15/32`, but selected exact/relaxed is only
`3/32`.

The drop from `5/32` to `3/32` also means expanded pairing should not become a
default setting until ranking is improved or candidate pruning prevents harmful
paired candidates from displacing better selections.

## Recommendation

Do not run a 64-sample expansion and do not make expanded pairing the default.

Next, analyze expanded-pairing oracle-miss ranking using this existing log,
especially the 7 oracle-only-pair misses. After that rank diagnosis, decide
whether to test a pair-aware ranking heuristic, a pair source penalty/bonus,
model-score tie-break, or candidate pruning. Do not change candidate generation
again until the ranking failure mode is clearer.
