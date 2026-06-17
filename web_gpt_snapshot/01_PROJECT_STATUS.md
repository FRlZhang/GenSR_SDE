# Project Status

## Current Goal

Improve sequence-level SDE recovery after establishing that the current unified fingerprint and role-token target are learnable by GenSR. Candidate generation plus fingerprint-distance reranking is implemented; drift/diffusion pairing now produces nonzero oracle hits, so the next step is diagnosing why the rerank score does not select those oracle candidates.

## Completed

- Added unified SDE fingerprint pipeline in `sde_fingerprint.py`.
- Added autonomous 1D SDE simulation helpers in `simulator_sde.py`.
- Added GenSR-compatible SDE dataset generation in `sde_dataset_generator.py`.
- Patched `train.py` to inject SDE samples through `EnvDataset.generate_sample`, correcting the earlier invalid `env.generate_sample` hook.
- Added `<DRIFT>` and `<DIFFUSION>` to the equation vocabulary.
- Added `sde_validation_probe.py` as the main fast train/eval/decoding harness.
- Added checkpoint save/load/eval-only support.
- Ran a 2000-step role-token probe with:

```text
token_top1=0.820002
token_top3=0.960833
token_top5=1.000000
greedy_sequence_exact=0.015625
```

- Ran constrained-beam eval from the same checkpoint; exact recovery improved only slightly (`1/32` on a 32-sample eval).
- Added constrained-beam n-best candidate generation and SDE fingerprint-distance reranking to `sde_validation_probe.py`.
- Verified reranking on a 4-sample eval-only checkpoint probe: 16 candidates attempted, 16 valid, 0 fingerprint failures, but `reranked_sequence_exact=0.000000` and `reranked_sequence_relaxed_no_constants=0.000000`.
- Added rerank debug output and oracle metrics. A 16-sample eval-only checkpoint probe produced 64 valid candidates and `rerank_oracle_sequence_exact=0.000000`, `rerank_oracle_sequence_relaxed_no_constants=0.000000`.
- Added grammar-constrained stochastic sampling with temperature, top-k, top-p, and multi-temperature support. A 16-sample eval-only checkpoint probe with beam+sampling produced 90 valid candidates and increased diversity to `rerank_unique_candidate_avg=5.625000`, `rerank_unique_drift_avg=2.000000`, `rerank_unique_diffusion_avg=4.875000`, but oracle exact/relaxed remained zero.
- Added drift/diffusion span pairing from existing candidates. A 16-sample eval-only checkpoint probe attempted 167 valid candidates, including 56 valid paired candidates, and reached `rerank_pair_oracle_sequence_exact=0.062500`, `rerank_pair_oracle_sequence_relaxed_no_constants=0.062500`; selected reranked exact/relaxed metrics remained zero.
- Added fingerprint segment distance diagnostics and `componentwise` reranking. On the same 16-sample checkpoint probe, both full and componentwise selected reranked exact/relaxed remained zero while oracle stayed 1/16. The inspected oracle candidate ranked 11 under full scoring and 8 under componentwise scoring, with the largest disadvantage in `multi_u0_moments`.
- Added global constant-grid reranking diagnostics. On the same 16-sample checkpoint probe, `constant_grid_componentwise` kept selected exact/relaxed at zero and oracle at 1/16, but improved the inspected oracle from rank 8 to rank 5 and reduced its distance from 5.919984 to 5.125097 with `best_constant=0.5`.
- Added top non-oracle diagnostics and score variants. `constant_grid_componentwise_no_multi_u0` was closest: selected exact/relaxed stayed zero and oracle stayed 1/16, but the inspected oracle moved to rank 2 with distance 1.507198, behind the top non-oracle at 1.505882 by only 0.001317.
- Added epsilon tie-breaking for near-equal rerank distances. With `constant_grid_componentwise_no_multi_u0`, `--rerank-tie-epsilon 0.005`, and either `--rerank-tie-break active_distance` or `--rerank-tie-break state_dependent_drift`, a 16-sample checkpoint probe reached the first selected reranked hit: exact/relaxed = 1/16.

## In Progress

- Transitioning from token recognition to full symbolic sequence recovery.
- Validating whether active-distance or state-dependent-drift tie-breaking is stable beyond the 16-sample diagnostic where it first selected the oracle.

## Next Steps

1. Re-run the strongest tie-break setting on 32/64 held-out samples.
2. Compare active-distance versus state-dependent-drift tie-breaking for false positives before making either the default rerank recipe.
3. Keep drift/diffusion pairing enabled and tune candidate pool size carefully because fingerprint recomputation is CPU-expensive.
4. Keep comparing greedy, constrained beam, reranked, and oracle metrics on the same held-out setup.
5. Only revisit fingerprint design after candidate diversity and scoring have been tested more thoroughly.

## Open Questions

- Should reranking use full unified fingerprint distance, componentwise distance, or short-time conditional moments?
- Should candidates be generated jointly as `<DRIFT> ... <DIFFUSION> ...`, or separately for drift/diffusion and paired later?
- How much error is decoder search versus template ambiguity?
- How well will current fingerprints transfer to real observed SDE data without multiple initial conditions?

## Known Issues

- Sequence-level recovery is weak even when teacher-forced token accuracy is strong.
- Grammar-constrained beam enforces validity better than correctness.
- First-pass reranking produces valid candidates but did not improve exact/relaxed recovery in a small checkpoint eval.
- Rerank oracle metrics were zero in a 16-sample checkpoint eval, so the immediate blocker is candidate diversity, not only rerank score selection.
- Stochastic grammar-constrained sampling increased unique candidates but still produced zero oracle hits in a 16-sample checkpoint eval.
- Drift/diffusion pairing produced nonzero oracle hits, but the current fingerprint-distance score still did not choose them in the 16-sample eval.
- Constant-grid componentwise reranking improved the inspected oracle candidate from rank 8 to 5, but still did not select it; fixed `CONSTANT=1.0` is part of the score problem, not the whole problem.
- Removing `multi_u0_moments` from the constant-grid score nearly selects the inspected oracle, but a non-oracle drift template still wins by 0.001317 due to weak-kernel distance.
- Tie-breaking can select the paired oracle in the inspected near-tie case, but this is only verified on 16 samples so far.
- The best checkpoint is outside the repo and may disappear.
- `environment.yml` is upstream/Linux-oriented; recent work used the local macOS `gensr` conda env.

## Best Checkpoint

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

## How To Resume

Read `00_WEB_GPT_INDEX.md`, this file, `02_RECENT_CHANGES.md`, `03_CURRENT_TASK.md`, and `04_COMMANDS.md`. If writing code, temporarily upload only the source files listed in `NEEDED_SOURCE_FILES.md` or in the index.
