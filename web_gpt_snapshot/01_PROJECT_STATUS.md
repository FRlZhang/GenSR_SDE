# Project Status

## Current Goal

Improve sequence-level SDE recovery after establishing that the current unified fingerprint and role-token target are learnable by GenSR. Candidate generation plus fingerprint-distance reranking is implemented; drift/diffusion pairing and role-wise constant-grid scoring now produce selected hits. The current strongest no-retraining selected-recovery setting remains `constant_grid_rolewise_no_multi_u0` with baseline pairing at selected exact/relaxed `5/32`. Expanded pairing raises oracle exact/relaxed to `15/32`, but selected exact/relaxed regressed to `3/32`, so expanded pairing is diagnostic-only for now. Residual-debug safety checks do not support scorer-side per-active-dimension normalization: the user-run expanded-pairing 16-sample smoke had oracle `5/16`, pair-oracle `4/16`, selected `0/16`, and normalization rescued `0/5` oracle-present misses. Expanded-pairing oracle-absent drift diversity diagnostics show the remaining `17/32` oracle-absent samples are all exact-drift misses with exact diffusion present. The completed drift-only smoke found exact drift in only `3/17` cases, all beam rank-30 nested-mul linear. JSON-only constant-folding diagnostics now show current expanded-pool folding projects `23/32` oracle coverage, exact-tail admission projects `18/32`, and drift-only tail plus folding projects `30/32`. A capped normalized admission diagnostic found `P2_one_per_family` preserves the `30/32` diagnostic ceiling while reducing target pair count from `1105` full admission to `340`; the reusable P2 hook and runner now reproduce those numbers with strict self-checks. The candidate coverage pipeline now exposes the validated P2 hook behind default-off `--normalized-drift-admission none|p2_one_per_family` and default source `beam`; JSON-only flag validation matches the prior P2 metrics with metric/schema/canonicalizer status `ok`, zero mismatches, and decision `A`. The explicit runtime coverage-only smoke now also matches: current expanded `15/32`, canonical expanded `23/32`, P2 normalized admission `30/32`, P2 pair count `340`, newly recovered `0,2,6,7,15,25,29`, remaining `16,28`, and `unsafe_collision_flag=False`. P2 scorer-ready sidecar logging now exists behind `--scorer-ready-json`; `p2_scorer_ready_candidates.json` contains target `y_to_fit` / fingerprint shape `[186, 1]`, 7/7 P2 oracle rows, 7/7 exact diffusion present, and parse-ready full candidate sequences. Validation decision `A`: next step can be offline semantic scoring of P2 admitted candidates. No semantic scoring or formal eval has been run from this result.

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
- Added checkpoint-backed drift/pairing coverage diagnostics in `scripts/analyze_drift_pairing_coverage.py` and `drift_pairing_coverage_diagnostics.md`. Drift-missing templates are mostly linear drift (6), sin drift (6), nested-mul linear drift (3), and polynomial-like drift (2). The full pairing rank diagnostic showed 5/6 pairing misses were blocked by `pair_drift_diffusion_candidates`, while sample 23 needed `pair_drift_topk >= 5`.
- Ran expanded-pairing candidate coverage with `pair_drift_topk=5`, `pair_diffusion_topk=6`, and `pair_drift_diffusion_candidates=32`. This rescued all six pairing-missing samples as pair-source full oracles and raised full-oracle coverage from `9/32` to `15/32`.
- The matching formal 32-sample eval reproduced oracle exact/relaxed `15/32` and pair oracle `8/32`, but selected exact/relaxed dropped to `3/32`. There were 15 oracle-present samples, 3 selected hits, and 12 selected misses, including 7 oracle-only-pair misses.
- Expanded-pairing oracle-miss ranking diagnostics show the misses are not mostly near ties: 2/12 have score gap `<=0.10`, and only 1/7 oracle-only-pair misses is that close. All 7 oracle-only-pair misses lose mainly on `active_kramers_moyal`; wrong pair candidates are selected in 8/12 misses.
- Wrong-pair pruning diagnostics found no safe oracle-preserving pruning rule. Pair-oracle source ranks extend to 15, wrong pair ranks overlap, and harmful structures such as mean-reverting or constant-drift pairs are also true oracle families in other samples.
- Active-residual trap diagnostics recomputed residual vectors for the 12 expanded-pairing selected misses only. Active distance favored the selected non-oracle in 12/12; 7/12 traps were outlier-dominated, 2/12 broad, and 3/12 mixed. Misleading active advantages leaned KM1/drift-like in 10/12 cases, and weak-kernel distance disagreed in 4/12.
- Offline active-residual ablation tested current logged score, clipped p90/p95, Huber p90, top-1/top-2 active removal, per-active-dimension normalization, and weak-only diagnostics. `per_active_dim_norm_plus_weak` flipped 6/12 misses and 4/7 oracle-only-pair misses with 0/3 debug top-2 hit harms, but it was calibrated only from miss residuals. Clipped/Huber/top-k variants flipped 3-5 misses but harmed one debug top-2 selected-hit check.
- Added optional `--rerank-residual-debug-json` to `sde_validation_probe.py` and `scripts/analyze_residual_debug_safety.py`. The baseline 16-sample smoke logged 201 candidates, 2 oracle samples, 1 selected hit, and 1 pair-oracle sample. The user-run expanded-pairing 16-sample smoke logged 350 candidates, 5 oracle samples, 4 pair-oracle samples, and 0 selected hits. Offline `per_active_dim_norm_plus_weak` rescued 0/5 oracle-present misses; oracle ranks under that score were 8, 6, 3, 3, and 11.
- Added `scripts/analyze_expanded_pairing_oracle_absent_drift_diversity.py` and wrote `expanded_pairing_oracle_absent_drift_diversity.md/json`. This parsed existing coverage JSON only. The remaining 17 expanded-pairing oracle-absent samples are all exact-drift misses; exact diffusion is present in all 17. Drift-family breakdown is linear 6, sin 6, nested-mul linear 3, polynomial-like 2. Missing drifts are absent from all logged sources, with nearest logged drifts collapsing toward over-nested or constant-heavy templates. Decision `A`: run one future diagnostic-only drift-only candidate diversity smoke.
- Added `scripts/analyze_drift_only_candidate_diversity.py` and `scripts/run_drift_only_candidate_diversity_smoke.sh`. The user completed the standalone smoke: exact drift appeared in `3/17` targets, all beam-only rank-30 nested-mul linear cases (`2`, `25`, `29`); sampling recovered `0/17`; linear `6/6`, sin `6/6`, and polynomial-like `2/2` remained exact-missing; exact diffusion was present in `17/17`.
- Added `scripts/analyze_drift_only_admission_and_constant_folding.py` and wrote `drift_only_admission_constant_folding_diagnostics.md/json`. This JSON-only helper found exact-tail admission projects only `+3` oracle coverage (`15/32` to `18/32`), while constant-chain folding matches `15/17` drift-only candidates and projects a diagnostic canonicalized ceiling of `30/32`. Linear and sin misses are all constant-over-nesting artifacts; polynomial-like samples `16` and `28` remain missing. Decision `B`: test candidate normalization / constant-chain folding before widening beam/top-k; do not run formal eval.
- Added `scripts/analyze_candidate_normalization_constant_folding.py` and wrote `candidate_normalization_constant_folding_diagnostics.md/json`. Current expanded-pool folding projects `23/32` oracle coverage (`+8`), exact-tail admission projects `18/32` (`+3`), and drift-only tail plus folding projects `30/32` (`+15`). Collision checks found no unsafe observed overmerge, and serialized drift candidates reduce `440 -> 278` (`36.8%`). Decision `B`: run one small coverage-only normalized drift-only admission diagnostic; do not run formal eval.
- Added `scripts/analyze_normalized_drift_only_admission_coverage.py` and wrote `normalized_drift_only_admission_coverage_diagnostics.md/json`. The diagnostic pool is current expanded pairing plus normalized drift-only tail candidates. It reproduced `15/32` current expanded, `23/32` current expanded canonical, `18/32` exact-tail, and `30/32` normalized admission. Newly recovered samples beyond current expanded canonical are `0,2,6,7,15,25,29`; only polynomial-like `16,28` remain missing. All admitted matches are beam-source with exact diffusion already present. Pair pressure is high: target pair estimates rise `370 -> 1105`; decision `B`.
- Added `scripts/analyze_capped_normalized_drift_admission.py` and wrote `capped_normalized_drift_admission_diagnostics.md/json`. Best practical policy is `P2_one_per_family`: it preserves `30/32` coverage and the same recovered targets as full normalized admission, while reducing target pair count from `1105` to `340` (`69.2%` reduction). Truth-aware upper bound is `30/32` at pair count `320`; sampling can be dropped; collision safety remains clean. Decision `A`.
- Added reusable coverage-only P2 admission hook `scripts/normalized_drift_admission.py` and runner `scripts/run_p2_normalized_admission_coverage.py`, producing `candidate_coverage_p2_normalized_admission.md/json`. The runner reproduces current expanded `15/32`, canonical expanded `23/32`, P2 normalized admission `30/32`, P2 pair count `340`, newly recovered `0,2,6,7,15,25,29`, remaining `16,28`, source `beam=15`, and sampling recovered canonical drifts `0`; self-checks passed.
- Added JSON-only reusable P2 hook cross-validation in `scripts/validate_p2_normalized_admission_hook.py`, producing `p2_normalized_admission_hook_validation.md/json`. It validated six reports, recomputed P2 through the reusable hook, and found all expected metrics consistent with no mismatches.
- Integrated the validated P2 hook into `scripts/analyze_candidate_coverage.py` behind default-off `--normalized-drift-admission none|p2_one_per_family`, with `--normalized-drift-admission-source beam`. Added `scripts/validate_candidate_coverage_p2_flag.py` and `candidate_coverage_p2_flag_validation.md/json`; the JSON-only validation reports metric cross-check `ok`, schema `ok`, canonicalizer safety `ok`, zero mismatches, and decision `A`. No model-backed candidate generation was run.
- Added `candidate_coverage_pair_expanded_p2_flag_runtime_report.md/json`. Runtime smoke decision is `D`: `/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log` is missing, so the coverage-only command was not run and `candidate_coverage_pair_expanded_p2_flag.md/json` was not produced.
- Regenerated `/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log`; it matches the recorded role-wise baseline metrics (`selected=5/32`, `oracle=9/32`, `pair_oracle=2/32`). Backed up this log and the surviving `/private/tmp/gensr_sde*.log/json` files to `experiment_logs/2026-06-24_tmp_gensr_sde_logs/`, which is ignored by git.
- Ran the explicit P2 candidate coverage flag smoke and produced `candidate_coverage_pair_expanded_p2_flag.md/json`. Runtime metrics match prior JSON-only validation exactly: current expanded `15/32`, canonical expanded `23/32`, P2 admission `30/32`, P2 pair count `340`, newly recovered `0,2,6,7,15,25,29`, remaining `16,28`, sampling recovered canonical drifts `0`, unsafe collision flag `False`. Updated `candidate_coverage_pair_expanded_p2_flag_runtime_report.md/json` to decision `A`.
- Added JSON-only P2 admitted-candidate scoring readiness diagnostics in `scripts/analyze_p2_admitted_candidate_scoring.py`, producing `p2_admitted_candidate_scoring_diagnostics.md/json`. The 7 P2 newly recovered samples (`0,2,6,7,15,25,29`) are all coverage-ready: exact diffusion is present and the P2 canonical oracle drift is beam-source rank 5 for each sample. Semantic scoring is blocked because the coverage JSON lacks target `y_to_fit` / fingerprint vectors or equivalent normalized eval samples with numeric constants. Decision `D`.
- Added default-off P2 scorer-ready sidecar logging to `scripts/analyze_candidate_coverage.py` with `--scorer-ready-json` and `--scorer-ready-target-samples`, plus `scripts/validate_p2_scorer_ready_logging.py`. The regenerated P2 coverage-only command wrote `p2_scorer_ready_candidates.json`; validation wrote `p2_scorer_ready_logging_validation.md/json` with decision `A`. The sidecar covers samples `0,2,6,7,15,25,29`, target fingerprint shape `[186, 1]`, 140 current-expanded rows, 105 P2-admitted rows, 7/7 P2 oracle rows, and no semantic scores.

## In Progress

- Transitioning from token recognition to full symbolic sequence recovery.
- Deciding the next offline semantic scoring diagnostic for P2 admitted candidates after scorer-ready sidecar logging validated with decision `A`. Do not treat the P2 coverage result as selected-rerank evidence or a production default change.
- Checkpoint-dependent candidate regeneration is unblocked: the 2000-step checkpoint is present in `/private/tmp` and backed up under `checkpoints/`, which is ignored by git.

## Next Steps

1. Do not make `active_distance` or `state_dependent_drift` the default based on the 16-sample hit; on 32 samples they matched no-tie baseline.
2. Keep `constant_grid_rolewise_no_multi_u0` active:weak 1:1 as the strongest current eval setting; do not make active-heavy weights the default.
3. Do not make expanded pairing the default and do not run 64-sample expansion: selected recovery regressed to `3/32`.
4. Do not add a simple pair-aware tie-break from the current evidence alone: most paired oracles lose by active+weak distance, not epsilon-scale ties.
5. Do not add pair pruning from current evidence: no oracle-preserving pruning rule was found.
6. Do not add robust/clipped/per-dimension normalized active scoring as a rerank mode. The expanded-pairing 16-sample residual-debug safety smoke had 5 oracle samples, 0 selected hits, and 0/5 offline miss rescues, so no formal 32-sample eval is justified for this scorer idea.
7. Do not add a robust-scoring rerank mode from the current offline ablation alone: mild variants rescued 0/4 and the only flip required aggressive clipping.
8. Keep comparing greedy, constrained beam, reranked, oracle, and oracle-miss metrics on the same held-out setup.
9. Keep `--normalized-drift-admission p2_one_per_family` as an explicit coverage-only diagnostic path; it now reproduces the `30/32` ceiling and pair count `340` in the actual candidate coverage pipeline.
10. Use `p2_scorer_ready_candidates.json` for the next offline semantic scoring diagnostic of P2 admitted candidates. Do not run formal eval or change defaults from sidecar readiness alone.
11. Do not run formal eval from drift-only or constant-folding diagnostics; they are oracle-coverage evidence, not selected-recovery evidence.
12. Only revisit fingerprint design after candidate diversity and scoring have been tested more thoroughly.

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
- Expanded pairing raises oracle coverage to `15/32`, but current scoring selected only `3/32`, so ranking of paired oracles is an immediate blocker.
- Expanded pairing leaves 17 oracle-absent samples; all 17 are exact-drift misses with exact diffusion present, so further oracle-ceiling work needs drift span diversity rather than pairing cap/top-k changes.
- Drift-only exact-tail admission is weak: only 3 rank-30 beam nested-mul linear cases are exact hits. Constant folding is much stronger but still diagnostic-only.
- Current expanded-pool constant folding alone is not enough (`23/32`), while drift-only tail plus folding projects `30/32`; this points to normalized drift-only admission rather than naive global beam widening.
- Full normalized drift-only admission is coverage-positive but high-pressure: canonicalization deduplicates the drift-only tail by `39.6%`, yet estimated target-sample pair count still rises by `198.6%`.
- Capped normalized admission is promising in coverage-only diagnostics: `P2_one_per_family` keeps `30/32` projected coverage with much lower pair pressure, but this still has no semantic fingerprint or selected-rerank validation.
- P2 scorer-ready sidecar logging is now available and validated, but selected rerank behavior is still untested.
- Constant-folding collision checks are token-structure-only; they did not flag unsafe observed overmerge, but semantic fingerprint validation and selected-rerank behavior remain untested.
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
