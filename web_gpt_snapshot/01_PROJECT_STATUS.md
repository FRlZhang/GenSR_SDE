# Project Status

## Current Goal

Improve sequence-level SDE recovery after establishing that the current unified fingerprint and role-token target are learnable by GenSR. Candidate generation plus fingerprint-distance reranking is implemented; drift/diffusion pairing and role-wise constant-grid scoring now produce selected hits. The current strongest no-retraining decoding setting is `constant_grid_rolewise_no_multi_u0` at selected exact/relaxed `5/32`, with oracle exact/relaxed still `9/32`.

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
- Ran the larger 32-sample eval-only tie-break validation from `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth`. Baseline no tie, `active_distance`, and `state_dependent_drift` all matched at selected exact/relaxed = 4/32, with oracle exact/relaxed = 9/32 and pair oracle exact/relaxed = 2/32. Since no tie-break exceeded baseline and each run was CPU-expensive, 64-sample validation was not run.
- Added oracle gap diagnostics to `sde_validation_probe.py`. The 32-sample no-tie rerun found 9 samples with oracle candidates, 4 selected hits, and 5 selected misses. Miss types: 2 same diffusion but wrong drift, 1 same drift but wrong diffusion, 2 both sides wrong, 0 constant-only mismatch. Best oracle sources were beam 6, sampling 1, pair 2; selected hits came from beam 3 and pair 1.
- Added `constant_grid_rolewise_no_multi_u0`. It searches separate drift and diffusion constants over the same constant grid. On the same 32-sample setup, active:weak 1:1 improved selected exact/relaxed to 5/32 and rescued 1/5 shared-constant misses; active:weak 2:1 and 1:0.5 both regressed to 3/32.
- Added targeted residual-vector diagnostics in `scripts/analyze_rolewise_residuals.py` and `rolewise_residual_vector_diagnostics.md`. The helper reproduces only samples 13, 19, 24, and 26 from the eval seed and scores selected/oracle role swaps. All four remaining misses were flagged as feature-scaling/fingerprint-ambiguity cases; sample 13 also has a model-score conflict.
- Added offline residual score ablation in `scripts/analyze_residual_score_ablation.py` and `rolewise_residual_score_ablation.md`. Mild robust/normalization variants rescued 0/4 remaining misses; only aggressive `clipped_l2_p90` flipped sample 19. No formal eval was run.
- Added candidate coverage diagnostics in `scripts/analyze_candidate_coverage.py` and `candidate_coverage_diagnostics.md`. The diagnostic reproduced the 9/32 full-oracle ceiling without fingerprint scoring. Among 23 oracle-absent samples, 17 have exact diffusion only and 6 have exact drift+diffusion separately but not paired.
- Added checkpoint-backed drift/pairing coverage diagnostics in `scripts/analyze_drift_pairing_coverage.py` and `drift_pairing_coverage_diagnostics.md`. Drift-missing templates are mostly linear drift (6), sin drift (6), nested-mul linear drift (3), and polynomial-like drift (2). The full pairing rank diagnostic shows 5/6 pairing misses are blocked by `pair_drift_diffusion_candidates`, while sample 23 needs `pair_drift_topk >= 5`.

## In Progress

- Transitioning from token recognition to full symbolic sequence recovery.
- Using candidate coverage diagnostics to raise the 9/32 oracle ceiling; drift span diversity is now the leading blocker, with pairing coverage/ranking secondary.
- Checkpoint-dependent candidate regeneration is unblocked: the 2000-step checkpoint is present in `/private/tmp` and backed up under `checkpoints/`, which is ignored by git.

## Next Steps

1. Do not make `active_distance` or `state_dependent_drift` the default based on the 16-sample hit; on 32 samples they matched no-tie baseline.
2. Keep `constant_grid_rolewise_no_multi_u0` active:weak 1:1 as the strongest current eval setting; do not make active-heavy weights the default.
3. Prioritize drift span diversity: 17/23 oracle-absent samples already contain the truth diffusion but miss the truth drift.
4. Improve pairing coverage/ranking for 6/23 oracle-absent samples where exact drift and exact diffusion appear separately but not as a full oracle: 5/6 are pair-cap misses and 1/6 needs a small drift-top-k increase.
5. Do not add a robust-scoring rerank mode from the current offline ablation alone: mild variants rescued 0/4 and the only flip required aggressive clipping.
6. Keep comparing greedy, constrained beam, reranked, oracle, and oracle-miss metrics on the same held-out setup.
7. Only revisit fingerprint design after candidate diversity and scoring have been tested more thoroughly.

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
- Tie-breaking can select the paired oracle in the inspected 16-sample near-tie case, but on 32 samples both tie-break modes matched the no-tie baseline instead of improving it.
- In the 32-sample oracle-miss diagnostics, selected non-oracles often beat oracle templates on active/weak distance. This points to score calibration / constant handling rather than a missing tie-break.
- Role-wise constants rescued one miss, but active/weak reweighting was brittle: 2:1 and 1:0.5 both dropped selected exact/relaxed to 3/32.
- Candidate generation remains a ceiling: only 9/32 samples had any oracle candidate.
- Constant-grid + pairing rerank is CPU-expensive; reserve 64-sample expansion for settings that first improve the 32-sample selected metrics.
- The best checkpoint is available at `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth` and backed up at `checkpoints/gensr_sde_role_token_2000/role_token_2000.pth`.
- `environment.yml` is upstream/Linux-oriented; recent work used the local macOS `gensr` conda env.

## Best Checkpoint

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

Persistent backup:

```text
checkpoints/gensr_sde_role_token_2000/role_token_2000.pth
```

Status: restored as of 2026-06-22; keep `checkpoints/` ignored and do not commit model weights.

## How To Resume

Read `00_WEB_GPT_INDEX.md`, this file, `02_RECENT_CHANGES.md`, `03_CURRENT_TASK.md`, and `04_COMMANDS.md`. If writing code, temporarily upload only the source files listed in `NEEDED_SOURCE_FILES.md` or in the index.
