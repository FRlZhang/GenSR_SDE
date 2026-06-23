# PROJECT_STATUS.md

## Current Goal

Improve sequence-level SDE recovery after establishing that the current unified
fingerprint and role-token target are learnable by GenSR. Candidate generation
plus fingerprint-distance reranking is now implemented in the validation probe;
drift/diffusion pairing and constant-grid scoring now produce nonzero selected
hits, but a 32-sample tie-break validation did not show incremental benefit
from the near-tie heuristics. Oracle-miss diagnostics show the immediate
selected-versus-oracle gap is mostly rerank score / constant handling, while
candidate generation remains the larger ceiling for samples with no oracle.
Role-wise constant-grid scoring rescued one selected-miss case and improved
selected exact/relaxed from `4/32` to `5/32`; active/weak reweighting variants
tested so far regressed. Expanded pairing raises offline full-oracle candidate
coverage from `9/32` to `15/32`, and a formal 32-sample eval reproduced that
oracle ceiling, but selected recovery dropped to `3/32`. Expanded pairing is
therefore useful diagnostically but should not become the default until
ranking/scoring of paired oracles improves. Targeted active-residual diagnostics
show the expanded-pairing misses are often `active_kramers_moyal` traps
dominated by a few KM1/drift-like residual dimensions, so the next scorer check
was performed offline. The strongest offline signal is per-active-dimension
normalization (`6/12` miss rescues, `4/7` oracle-only-pair rescues), but it was
calibrated only on miss residuals; clipped/Huber/top-k variants rescued several
misses while harming one debug top-2 selected-hit check. This does not justify a
new rerank mode or formal eval yet. A residual-debug JSON export now exists for
full candidate-pool offline safety checks. The first 8-sample smoke validated
logging only; a follow-up 16-sample baseline-pairing smoke logged 201
candidates with 2 oracle samples and 1 selected hit. Offline
per-active-dimension normalization preserved that hit and harmed none, but
rescued 0/1 oracle-present misses. The user then ran the prepared
expanded-pairing 16-sample smoke, which logged 350 candidates and increased
oracle availability to 5/16, with 4/16 pair-oracle samples, but selected
recovery dropped to 0/16 and offline normalization rescued 0/5 oracle-present
misses. Therefore scorer-side normalization is not supported by current safety
evidence; do not add `per_active_dim_norm_plus_weak` as a rerank mode or run a
formal 32-sample eval for it. Expanded-pairing oracle-absent drift diversity
diagnostics now show that all remaining `17/32` oracle-absent samples are
exact-drift misses with exact diffusion already present; pairing-missing and
diffusion-missing are both `0/17`. The next direction should be one
diagnostic-only drift-only candidate diversity smoke that logs drift-span
ranks/sources before any rerank, formal-eval, or candidate-generation default
change. A helper for that smoke now exists, but the first Codex-run attempt
exceeded the 10-minute budget and was interrupted before sample metrics were
produced. Run the standalone script before deciding whether the exact drifts
are hidden in wider tails or absent from current drift-only beam/sampling. The
user-run standalone smoke found exact drifts in only `3/17` drift-only
candidates, all beam rank-30 nested-mul linear cases, while exact diffusion was
already present for `17/17`. A follow-up JSON-only admission and
constant-folding diagnostic found that exact-tail admission alone projects only
`+3` oracle coverage (`18/32`), but narrow constant-chain folding matches
`15/17` drift-only candidates and explains all `12/12` linear/sin exact misses;
the two polynomial-like cases remain missing. A helper-only candidate
normalization diagnostic now separates current expanded-pool normalization from
drift-only tail admission: current expanded-pool folding projects `23/32`
oracle coverage (`+8`), exact-tail admission projects `18/32` (`+3`), and
drift-only tail plus folding projects `30/32` (`+15`). Observed collisions are
constant-chain-only, with no unsafe `pow2`/linear or `sin`/linear overmerge,
and serialized drift candidates reduce from `440` to `278` (`36.8%`). Decision
`B`: run one small coverage-only smoke/admission diagnostic with normalized
drift-only candidates before widening beam/top-k, and do not run formal eval.

## Last Updated

2026-06-23

## Completed

- Added a unified SDE fingerprint pipeline in [sde_fingerprint.py](/Users/lzhang/Documents/GenSR_SDE/sde_fingerprint.py).
- Added 1D autonomous SDE simulation helpers in [simulator_sde.py](/Users/lzhang/Documents/GenSR_SDE/simulator_sde.py).
- Added GenSR-compatible SDE dataset generation in [sde_dataset_generator.py](/Users/lzhang/Documents/GenSR_SDE/sde_dataset_generator.py).
- Patched [train.py](/Users/lzhang/Documents/GenSR_SDE/train.py) to inject SDE
  samples through `EnvDataset.generate_sample` instead of the earlier invalid
  hook on `env.generate_sample`.
- Added explicit `<DRIFT>` and `<DIFFUSION>` tokens to the equation vocabulary.
- Added a validation probe with held-out metrics in
  [sde_validation_probe.py](/Users/lzhang/Documents/GenSR_SDE/sde_validation_probe.py).
- Added checkpoint save/load/eval-only support to the validation probe.
- Ran a 2000-step checkpointed role-token probe with:
  `token_top1=0.820002`, `token_top3=0.960833`, `token_top5=1.000000`.
- Ran an eval-only constrained-beam test from the 2000-step checkpoint; exact
  recovery improved only slightly (`1/32`).
- Added diverse constrained-beam candidate generation and SDE fingerprint-distance
  reranking to [sde_validation_probe.py](/Users/lzhang/Documents/GenSR_SDE/sde_validation_probe.py).
- Verified reranking on a 4-sample eval-only checkpoint probe:
  `rerank_candidates_attempted=16`, `rerank_candidates_valid=16`, but
  `reranked_sequence_exact=0.000000` and
  `reranked_sequence_relaxed_no_constants=0.000000`.
- Added rerank debug output and oracle candidate metrics. A 16-sample eval-only
  checkpoint probe with 64 valid candidates reported
  `rerank_oracle_sequence_exact=0.000000` and
  `rerank_oracle_sequence_relaxed_no_constants=0.000000`, indicating the
  current constrained-beam candidate pool did not contain the target templates.
- Added grammar-constrained stochastic sampling candidates with temperature,
  top-k, top-p, and multi-temperature support. A 16-sample eval-only checkpoint
  probe with beam+sampling attempted 90 valid candidates and increased diversity
  to `rerank_unique_candidate_avg=5.625000`,
  `rerank_unique_drift_avg=2.000000`, and
  `rerank_unique_diffusion_avg=4.875000`, but oracle exact/relaxed remained 0.
- Added drift/diffusion span pairing from existing beam+sampling candidates. A
  16-sample eval-only checkpoint probe attempted 167 valid candidates, including
  56 valid paired candidates, and reached
  `rerank_pair_oracle_sequence_exact=0.062500` and
  `rerank_pair_oracle_sequence_relaxed_no_constants=0.062500`; selected
  reranked exact/relaxed metrics remained 0.
- Added fingerprint segment distance diagnostics and a `componentwise` rerank
  score. On the same 16-sample checkpoint probe, both `full_fingerprint` and
  `componentwise` kept selected reranked exact/relaxed at 0 while oracle stayed
  1/16. The observed oracle candidate ranked 11 under `full_fingerprint` and 8
  under `componentwise`; all fingerprint segments scored it worse than the top
  candidate, with the largest gap in `multi_u0_moments`.
- Added global constant-grid reranking diagnostics. On the same 16-sample
  checkpoint probe, `constant_grid_componentwise` kept selected exact/relaxed at
  0 and oracle at 1/16, but improved the inspected oracle candidate from rank 8
  to rank 5 and reduced its distance from 5.919984 to 5.125097 with
  `best_constant=0.5`.
- Added top non-oracle template diagnostics and constant-grid score variants.
  In the same 16-sample probe, `constant_grid_componentwise_no_multi_u0` was the
  closest variant: selected exact/relaxed still stayed 0 and oracle stayed 1/16,
  but the inspected oracle moved to rank 2 with distance 1.507198, behind a
  non-oracle at 1.505882 by only 0.001317.
- Added epsilon tie-breaking for near-equal rerank distances. With
  `constant_grid_componentwise_no_multi_u0`, `--rerank-tie-epsilon 0.005`, and
  either `--rerank-tie-break active_distance` or
  `--rerank-tie-break state_dependent_drift`, a 16-sample checkpoint probe
  achieved the first selected reranked hit:
  `reranked_sequence_exact=0.062500` and
  `reranked_sequence_relaxed_no_constants=0.062500`.
- Validated the near-tie settings on a 32-sample eval-only checkpoint probe
  using the same 2000-step checkpoint and no retraining. Baseline no tie,
  `active_distance`, and `state_dependent_drift` all produced the same selected
  recovery: `reranked_sequence_exact=0.125000` and
  `reranked_sequence_relaxed_no_constants=0.125000`, with oracle exact/relaxed
  still at `0.281250`. Since neither tie-break exceeded baseline, the 64-sample
  extension was not triggered.
- Added oracle gap diagnostics to [sde_validation_probe.py](/Users/lzhang/Documents/GenSR_SDE/sde_validation_probe.py)
  and ran the 32-sample no-tie setting again. The probe found 9 samples with an
  oracle candidate, 4 selected hits, and 5 selected misses. Best oracle source
  counts were beam 6, sampling 1, pair 2; selected hits came from beam 3 and
  pair 1. Miss types were: 2 same diffusion but wrong drift, 1 same drift but
  wrong diffusion, 2 both drift and diffusion wrong, 0 constant-only mismatch.
  Four misses had oracle candidates from beam/sampling but were scored lower
  than non-oracles, and one oracle appeared only from pairing.
- Added `constant_grid_rolewise_no_multi_u0`, which keeps the active+weak
  no-`multi_u0` score but searches separate shared constants for drift and
  diffusion roles. On the same 32-sample eval-only setup it improved selected
  exact/relaxed from `4/32` to `5/32` while the oracle ceiling stayed `9/32`.
  It rescued the same-drift/wrong-diffusion miss at sample 17 by using
  drift constant `1.0` and diffusion constant `0.5`. Active-heavy `2:1` and
  weak-downweighted `1:0.5` variants both regressed to `3/32`.
- Added a targeted residual-vector diagnostic helper in
  [scripts/analyze_rolewise_residuals.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_rolewise_residuals.py)
  and wrote
  [rolewise_residual_vector_diagnostics.md](/Users/lzhang/Documents/GenSR_SDE/rolewise_residual_vector_diagnostics.md).
  The helper does not run a formal 32-sample eval; it reproduces the four known
  role-wise misses (`13`, `19`, `24`, `26`) from the eval seed and scores only
  selected/oracle role swaps. The residual vectors flagged all four misses as
  feature-scaling/fingerprint-ambiguity cases, with sample 13 also carrying a
  model-score conflict.
- Added an offline residual score-ablation helper in
  [scripts/analyze_residual_score_ablation.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_residual_score_ablation.py)
  and wrote
  [rolewise_residual_score_ablation.md](/Users/lzhang/Documents/GenSR_SDE/rolewise_residual_score_ablation.md).
  Mild robust/normalization variants rescued `0/4` remaining role-wise misses;
  only the aggressive `clipped_l2_p90` variant flipped sample 19. This does not
  justify a formal 32-sample robust-scoring eval yet.
- Added a candidate coverage diagnostic helper in
  [scripts/analyze_candidate_coverage.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_candidate_coverage.py)
  and wrote
  [candidate_coverage_diagnostics.md](/Users/lzhang/Documents/GenSR_SDE/candidate_coverage_diagnostics.md).
  It reruns candidate generation only, without fingerprint scoring, retraining,
  or 64-sample expansion. It reproduced the `9/32` full-oracle ceiling and
  found the 23 oracle-absent samples split into 17 diffusion-only coverage cases
  and 6 exact-drift/exact-diffusion separate-but-unpaired cases. No sample had
  drift-only or neither-side coverage.
- Added a drift/pairing coverage helper in
  [scripts/analyze_drift_pairing_coverage.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_drift_pairing_coverage.py)
  and wrote
  [drift_pairing_coverage_diagnostics.md](/Users/lzhang/Documents/GenSR_SDE/drift_pairing_coverage_diagnostics.md).
  After restoring the 2000-step checkpoint, candidate coverage was regenerated
  with a JSON sidecar and the full rank diagnostic completed. The 17
  drift-missing cases are linear drift 6, sin drift 6, nested linear drift 3,
  and polynomial-like drift 2. For the 6 pairing-missing cases, 5 already have
  exact drift and diffusion inside the current pair top-k but are blocked by
  `pair_drift_diffusion_candidates`; sample 23 needs `pair_drift_topk` raised
  from 4 to at least 5.
- Ran an expanded-pairing candidate coverage probe with `pair_drift_topk=5`,
  `pair_diffusion_topk=6`, and `pair_drift_diffusion_candidates=32`. This was
  candidate-generation coverage only, not a formal rerank eval. It rescued all
  6 pairing-missing samples (`5`, `8`, `11`, `23`, `30`, `31`) as full oracle
  candidates from the `pair` source and raised the full-oracle ceiling from
  `9/32` to `15/32`. The remaining 17 oracle-absent samples are all still
  drift-missing with exact diffusion present.
- Ran one formal 32-sample eval-only probe with the current strongest scorer
  (`constant_grid_rolewise_no_multi_u0`) and expanded pairing. Oracle
  exact/relaxed rose from `9/32` to `15/32`, and pair oracle rose from `2/32`
  to `8/32`, matching the coverage diagnostic. Selected exact/relaxed fell
  from the role-wise baseline `5/32` to `3/32`, with 15 oracle-present samples,
  3 selected oracle hits, and 12 selected misses. Of the misses, 7 were
  oracle-only-pair cases and 5 were beam/sampling score misses; role-wise
  constants rescued 0 listed misses in this expanded run.
- Added expanded-pairing oracle-miss ranking diagnostics in
  [scripts/analyze_expanded_pairing_misses.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_expanded_pairing_misses.py)
  and
  [expanded_pairing_oracle_miss_ranking_diagnostics.md](/Users/lzhang/Documents/GenSR_SDE/expanded_pairing_oracle_miss_ranking_diagnostics.md).
  The 12 selected misses were not mostly near ties: only 2/12 had score gap
  `<=0.10`, and only 1/7 oracle-only-pair misses was that close. For all 7
  oracle-only-pair misses, the selected non-oracle's advantage was dominated by
  `active_kramers_moyal`. Wrong pair candidates were selected in 8/12 misses,
  so expanded pairing adds useful oracles and harmful high-ranking pair
  candidates at the same time.
- Added wrong-pair pruning diagnostics in
  [scripts/analyze_expanded_pair_pruning.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_expanded_pair_pruning.py)
  and
  [expanded_pairing_wrong_pair_pruning_diagnostics.md](/Users/lzhang/Documents/GenSR_SDE/expanded_pairing_wrong_pair_pruning_diagnostics.md).
  The 8 wrong selected pair cases are parse-valid, structurally plausible, and
  overlap with true pair-oracle families. Pair source-rank pruning is unsafe
  because preserving all 8 pair-oracle samples requires keeping pair source
  ranks up to 15, which also keeps the observed wrong pair selections. No safe
  pruning criterion was found from existing logs/JSON.
- Added active-kramers-moyal residual trap diagnostics in
  [scripts/analyze_expanded_pair_active_residuals.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_expanded_pair_active_residuals.py)
  and
  [expanded_pairing_active_residual_trap_diagnostics.md](/Users/lzhang/Documents/GenSR_SDE/expanded_pairing_active_residual_trap_diagnostics.md).
  The helper did not run model decoding, candidate regeneration, formal eval,
  retraining, or scorer changes; it recomputed residual vectors for the 12
  expanded-pairing selected misses only. Active distance favored the selected
  non-oracle in all 12 misses; 7/12 traps were outlier-dominated, 2/12 broad,
  and 3/12 mixed. Misleading active residual advantages leaned KM1/drift-like in
  10/12 cases. Weak-kernel distance disagreed in 4/12 cases. This supports one
  future offline robust/clipped active-residual ablation on existing expanded
  candidates, but does not justify a formal eval or new rerank mode yet.
- Added offline active-residual ablations in
  [scripts/analyze_expanded_pair_active_ablation.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_expanded_pair_active_ablation.py)
  and
  [expanded_pairing_active_residual_ablation.md](/Users/lzhang/Documents/GenSR_SDE/expanded_pairing_active_residual_ablation.md).
  This used existing residual/candidate logs only; it did not run formal eval,
  model decoding, candidate regeneration, retraining, grids, or scorer changes.
  The best offline variant, `per_active_dim_norm_plus_weak`, flipped 6/12
  selected misses and 4/7 oracle-only-pair misses without harming the limited
  debug top-2 selected-hit check. However, the normalization is calibrated only
  from miss residuals. Clipped/Huber/top-k active variants flipped 3-5 misses
  but harmed one debug top-2 selected-hit check. Decision: do not add a rerank
  mode or run a formal eval yet; if scorer work continues, add richer residual
  logging in a small 8- or 16-sample smoke first.
- Added optional residual debug JSON export to
  [sde_validation_probe.py](/Users/lzhang/Documents/GenSR_SDE/sde_validation_probe.py)
  via `--rerank-residual-debug-json`, plus
  [scripts/analyze_residual_debug_safety.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_residual_debug_safety.py).
  With the flag absent, rerank behavior is unchanged. An 8-sample eval-only
  smoke from the 2000-step checkpoint logged 101 valid candidates with
  candidate labels, source/rank metadata, constants, active residual vectors
  (18 dims), and weak residual vectors (96 dims), but it had 0 oracle candidates
  and 0 selected hits. A follow-up 16-sample smoke logged 201 valid candidates,
  2 oracle samples, 1 selected hit, and 1 pair-oracle sample. The offline
  `per_active_dim_norm_plus_weak` safety parse preserved the selected hit and
  harmed 0, but rescued 0/1 oracle-present selected misses. This validates the
  logging path and gives limited safety evidence, but still does not justify a
  formal eval or new rerank mode.
- Recorded the user-run expanded-pairing 16-sample residual-debug smoke and
  safety parse. Expanded pairing logged 350 valid candidates, raised oracle
  availability from 2/16 to 5/16 and pair-oracle availability from 1/16 to
  4/16, but selected recovery dropped from 1/16 to 0/16. The offline
  `per_active_dim_norm_plus_weak` safety parse produced 0 offline selected
  hits, 0 selected-miss rescues, and 8 offline selected pair candidates. The
  `selected-hit harms=0` count is not strong safety evidence because there were
  no expanded-pairing selected hits to harm. Oracle ranks under the offline
  score were sample 5 rank 8, sample 8 rank 6, sample 11 rank 3, sample 12
  rank 3, and sample 13 rank 11.
- Added an offline drift-diversity diagnostic helper in
  [scripts/analyze_expanded_pairing_oracle_absent_drift_diversity.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_expanded_pairing_oracle_absent_drift_diversity.py)
  and wrote
  [expanded_pairing_oracle_absent_drift_diversity.md](/Users/lzhang/Documents/GenSR_SDE/expanded_pairing_oracle_absent_drift_diversity.md).
  The helper parses existing coverage JSON only; it did not run
  `sde_validation_probe.py`, candidate regeneration, fingerprint simulation,
  formal eval, 64-sample eval, grids, retraining, scorer changes, or
  candidate-generation changes. The 17 expanded-pairing oracle-absent samples
  are all exact-drift misses with exact diffusion already present. Drift-family
  breakdown is linear 6, sin 6, nested-mul linear 3, polynomial-like 2. Exact
  drift is absent from all logged sources, while nearest logged drifts are
  often nearby over-nested or constant-heavy templates; decision `A`
  recommends one future diagnostic-only drift-only candidate diversity smoke.
- Added
  [scripts/analyze_drift_only_candidate_diversity.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_drift_only_candidate_diversity.py)
  and
  [scripts/run_drift_only_candidate_diversity_smoke.sh](/Users/lzhang/Documents/GenSR_SDE/scripts/run_drift_only_candidate_diversity_smoke.sh)
  for a model-backed drift-only candidate diversity smoke over the 17
  expanded-pairing oracle-absent samples. The helper compiles and is
  diagnostic-only: it loads the current checkpoint, generates drift-only
  constrained beam/sampling candidates, and writes
  [drift_only_candidate_diversity_smoke.md](/Users/lzhang/Documents/GenSR_SDE/drift_only_candidate_diversity_smoke.md)
  /
  [drift_only_candidate_diversity_smoke.json](/Users/lzhang/Documents/GenSR_SDE/drift_only_candidate_diversity_smoke.json).
  The first Codex-run attempt exceeded the 10-minute budget and was interrupted,
  but the user later ran the standalone script successfully. The completed
  smoke found exact drift in `3/17` targets, all beam-only rank-30 nested-mul
  linear cases (`2`, `25`, `29`); sampling recovered `0/17`, exact drift stayed
  missing for all linear, sin, and polynomial-like cases, and exact diffusion
  was already present for all `17/17` targets.
- Added
  [scripts/analyze_drift_only_admission_and_constant_folding.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_drift_only_admission_and_constant_folding.py)
  and wrote
  [drift_only_admission_constant_folding_diagnostics.md](/Users/lzhang/Documents/GenSR_SDE/drift_only_admission_constant_folding_diagnostics.md)
  /
  [drift_only_admission_constant_folding_diagnostics.json](/Users/lzhang/Documents/GenSR_SDE/drift_only_admission_constant_folding_diagnostics.json).
  This was JSON-only; it did not run model decoding, candidate generation,
  fingerprint simulation, reranking, formal eval, 64-sample eval, grids,
  retraining, scorer changes, rerank-mode changes, checkpoint/data changes,
  target-format changes, fingerprint changes, or production default changes.
  Exact-tail admission projects only `+3` full-oracle coverage, from `15/32` to
  `18/32`, and all exact hits are beam rank 30. Constant-chain folding matches
  `15/17` drift-only candidates, including all linear and sin misses, for a
  diagnostic-only canonicalized ceiling of `30/32`; polynomial-like samples
  `16` and `28` remain missing. Decision `B`: test candidate normalization /
  constant-chain folding before widening beam/top-k, and do not run formal eval.
- Added
  [scripts/analyze_candidate_normalization_constant_folding.py](/Users/lzhang/Documents/GenSR_SDE/scripts/analyze_candidate_normalization_constant_folding.py)
  and wrote
  [candidate_normalization_constant_folding_diagnostics.md](/Users/lzhang/Documents/GenSR_SDE/candidate_normalization_constant_folding_diagnostics.md)
  /
  [candidate_normalization_constant_folding_diagnostics.json](/Users/lzhang/Documents/GenSR_SDE/candidate_normalization_constant_folding_diagnostics.json).
  This helper parsed existing JSON only and applied the same narrow
  constant-chain canonicalizer at the candidate-normalization level. Current
  expanded-pool folding projects `23/32` oracle coverage (`+8`, samples
  `1,3,4,9,10,14,20,22`), exact-tail admission projects `18/32` (`+3`,
  samples `2,25,29`), and drift-only tail plus folding projects `30/32`
  (`+15`, all targets except polynomial-like samples `16` and `28`). Collision
  checks found no unsafe `pow2`/linear or `sin`/linear overmerge, and
  serialized drift candidates reduce `440 -> 278` (`36.8%`). Decision `B`: run
  one small coverage-only normalized drift-only admission diagnostic; do not run
  formal eval.
- Logged validated experiments in
  [SDE_TRAINING_REPORT.md](/Users/lzhang/Documents/GenSR_SDE/SDE_TRAINING_REPORT.md).

## In Progress

- Transitioning from "can the model learn the fingerprint?" to "can we decode
  the right symbolic sequence?".
- Planning one small coverage-only normalized drift-only admission diagnostic
  before widening drift beam/top-k or changing production candidate-generation
  defaults. Residual-debug safety checks do not support continuing scorer-side
  normalization.
- The 2000-step role-token checkpoint has been restored in `/private/tmp` and
  backed up under `checkpoints/`; model weights remain ignored by git.

## Next Steps

1. Do not make `active_distance` or `state_dependent_drift` the default based
   on the 16-sample hit; on 32 samples they matched, but did not beat, no tie.
2. Keep `constant_grid_rolewise_no_multi_u0` as the strongest current
   no-retraining decoding setting (`5/32` selected, `9/32` oracle), but do not
   continue broad active/weak weight search because the two tested variants
   regressed.
3. Use the residual-vector diagnostic output to decide whether feature
   normalization/scaling needs a targeted check before adding any new scoring
   heuristic. The four remaining role-wise misses are not near-ties and still
   favor selected non-oracles on aggregate active+weak residuals.
4. Do not add a robust-scoring rerank mode based on the offline ablation alone:
   mild variants rescued `0/4`, and the only flip required aggressive clipping.
5. Do not make expanded pairing the default and do not run 64-sample expansion:
   formal selected exact/relaxed regressed to `3/32` despite oracle rising to
   `15/32`.
6. Do not add a simple pair-aware tie-break from the current evidence alone:
   most paired oracles lose by active+weak distance, not epsilon-scale ties.
7. Do not add pair pruning from the current evidence: no oracle-preserving
   pruning rule was found from existing logs/JSON.
8. Do not add robust/clipped/per-dimension normalized active scoring as a rerank
   mode from the offline ablation or residual-debug safety smokes. The
   expanded-pairing 16-sample smoke had 5 oracle samples, 0 selected hits, and
   0/5 offline miss rescues, so do not run a formal 32-sample eval for this
   scorer idea.
9. Prefer one small coverage-only normalized drift-only admission diagnostic
   before widening beam/top-k. Current expanded-pool canonicalization alone
   projects `23/32`, while drift-only tail plus canonicalization projects
   `30/32`.
10. Do not run formal eval from the drift-only or constant-folding diagnostics;
   they are oracle-coverage diagnostics, not selected-recovery evidence.
11. Keep drift/diffusion pairing enabled and tune candidate pool size carefully
   because fingerprint recomputation is CPU-expensive.
12. Reuse the saved 2000-step checkpoint for decoding experiments instead of
   retraining.
13. Compare greedy, constrained beam, reranked, and oracle candidate metrics on the same
   held-out evaluation setup.
14. Only revisit fingerprint design after candidate diversity and scoring have
   been tested more thoroughly.

## Open Questions

- What reranking score is best: full unified fingerprint distance, componentwise
  distance, or a short-time conditional-moment-only score?
- Should candidate generation stay joint over
  `<DRIFT> ... <DIFFUSION> ...`, or should drift/diffusion templates be reranked
  separately and then paired?
- How much of the remaining error comes from decoder search versus template
  ambiguity in the current symbolic vocabulary?
- How well will the current fingerprint transfer from simulated training data to
  real observed SDE data, where multi-initial-condition information may not be
  available?

## Important Decisions

- Keep the current unified fingerprint route; do not restart fingerprint design
  before exhausting decoding improvements.
- Use explicit role tokens `<DRIFT>` and `<DIFFUSION>` as the default symbolic
  target format.
- Do not use two fully independent drift/diffusion decoders as the default
  architecture; they were not sample-efficient in short CPU runs.
- Treat `sde_validation_probe.py` as the main experiment harness for fast
  iteration and checkpoint reuse.
- Treat the 2000-step role-token checkpoint as the current best resume point for
  decoding experiments.

## Known Issues

- Sequence-level recovery is still weak even when teacher-forced token accuracy
  is strong.
- Grammar-constrained beam helps only slightly; it enforces validity better than
  correctness.
- First-pass fingerprint reranking can produce valid candidates but did not
  improve exact/relaxed recovery in a small 4-sample checkpoint eval.
- Rerank oracle metrics were also zero in a 16-sample checkpoint eval, so the
  immediate blocker is candidate diversity rather than only rerank scoring.
- Stochastic grammar-constrained sampling increased unique candidates but still
  did not produce oracle hits in a 16-sample checkpoint eval.
- Drift/diffusion pairing produced nonzero oracle hits, but the current
  fingerprint-distance score still did not choose them in the 16-sample eval.
- Constant-grid componentwise reranking improved the inspected oracle candidate
  from rank 8 to 5, but still did not select it; fixed `CONSTANT=1.0` is part of
  the score problem, not the whole problem.
- Removing `multi_u0_moments` from the constant-grid score nearly selects the
  oracle, but a non-oracle template still wins by 0.001317 due to a slightly
  better weak-kernel distance.
- Tie-breaking selected the paired oracle in one 16-sample near-tie diagnostic,
  but on the 32-sample validation both tie-break modes matched the no-tie
  baseline instead of improving it.
- In the 32-sample oracle-miss diagnostics, selected candidates often beat the
  oracle on the active/weak score even when the oracle template is present. This
  points to score calibration / constant handling, not a missing tie-break.
- Role-wise constants can rescue near misses, but only one of the five shared
  selected misses was rescued. The remaining misses still favor non-oracles in
  active/weak distance even after oracle scores improve.
- Expanded pairing raises the oracle ceiling to `15/32`, but current scoring
  selected only `3/32`, so scorer/ranking is the immediate blocker for the
  expanded pool.
- Drift-only exact-tail admission is weak: only 3 rank-30 beam nested-mul
  linear cases are exact hits. Constant folding is stronger but still
  diagnostic-only.
- Current expanded-pool constant folding alone is not enough (`23/32`), while
  drift-only tail plus folding projects `30/32`; this points to normalized
  drift-only admission rather than naive global beam widening.
- Constant-folding collision checks are token-structure-only; they did not flag
  unsafe observed overmerge, but semantic fingerprint validation and
  selected-rerank behavior remain untested.
- The 32-sample constant-grid + pairing probe is CPU-expensive, so 64-sample
  expansion should be reserved for settings that first improve the 32-sample
  selected metrics.
- The main checkpoint is expected at
  `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth` and is backed up
  at `checkpoints/gensr_sde_role_token_2000/role_token_2000.pth`; `checkpoints/`
  is ignored by git.
- `environment.yml` is upstream and Linux-oriented; the recent local experiments
  were run from the existing `gensr` conda environment on macOS.
- Keep future git changes grouped by experiment stage so decoding changes can
  be compared against this checkpointed role-token baseline cleanly.

## How To Resume

Read these files first:

1. [README.md](/Users/lzhang/Documents/GenSR_SDE/README.md)
2. [AGENTS.md](/Users/lzhang/Documents/GenSR_SDE/AGENTS.md)
3. [PROJECT_STATUS.md](/Users/lzhang/Documents/GenSR_SDE/PROJECT_STATUS.md)
4. [SDE_TRAINING_REPORT.md](/Users/lzhang/Documents/GenSR_SDE/SDE_TRAINING_REPORT.md)
5. `git -C /Users/lzhang/Documents/GenSR_SDE status --short`

Current best checkpoint:

- `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth`
- persistent backup:
  `/Users/lzhang/Documents/GenSR_SDE/checkpoints/gensr_sde_role_token_2000/role_token_2000.pth`

If that checkpoint still exists, start from eval-only decoding work:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --eval-only \
  --load-checkpoint /private/tmp/gensr_sde_role_token_2000/role_token_2000.pth \
  --eval-samples 32 \
  --batch-size 8 \
  --n-paths 800 \
  --active-paths 800 \
  --n-steps 60 \
  --max-generated-len 40 \
  --min-generated-len 8 \
  --constrained-beam-size 8
```

If the checkpoint is missing, regenerate it with:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
/opt/miniconda3/envs/gensr/bin/python3 sde_validation_probe.py \
  --train-steps 2000 \
  --eval-samples 64 \
  --batch-size 8 \
  --print-freq 250 \
  --n-paths 800 \
  --active-paths 800 \
  --n-steps 60 \
  --max-generated-len 40 \
  --min-generated-len 8 \
  --save-checkpoint /private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

First verification commands after any code change:

- `PYTHONPYCACHEPREFIX=/private/tmp/gensr_pycache /opt/miniconda3/envs/gensr/bin/python3 -m py_compile sde_validation_probe.py`
- `git -C /Users/lzhang/Documents/GenSR_SDE diff --check`
