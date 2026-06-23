# Recent Changes

## Last Codex Workflow

Codex added an offline drift-diversity diagnostic for the 17 expanded-pairing
oracle-absent samples:

```text
scripts/analyze_expanded_pairing_oracle_absent_drift_diversity.py
expanded_pairing_oracle_absent_drift_diversity.md
expanded_pairing_oracle_absent_drift_diversity.json
```

This parsed existing coverage JSON only. It did not run `sde_validation_probe.py`,
candidate regeneration, model decoding, fingerprint simulation, formal
32-sample eval, 64-sample eval, grids, retraining, scorer changes, or
candidate-generation changes.

Summary:

```text
expanded full_oracle_present=15/32
expanded oracle_absent=17/32
exact_drift_missing=17/17
pairing_missing=0/17
diffusion_missing=0/17
exact_diffusion_present=17/17
nearest_drift_right_family=9/17
decision=A
```

Drift-family breakdown:

```text
linear drift=6
sin drift=6
nested-mul linear drift=3
polynomial-like drift=2
constant drift=0
other/unknown=0
```

Interpretation: expanded pairing rescued all six previous pairing-missing
cases, so the remaining oracle ceiling problem is drift span generation. Exact
drift is absent from all logged sources for the remaining 17 samples; nearest
logged drifts are often nearby over-nested or constant-heavy templates. Exact
diffusion is already present for every remaining oracle-absent sample. Existing
logs do not contain unlogged beam/sampling tail ranks, token logits/entropy, or
grammar rejection counts, so they cannot prove whether the exact drift is below
an unlogged top-k tail versus absent from the model/grammar distribution.

Recommendation: run one future diagnostic-only drift-only candidate diversity
smoke that logs drift-span ranks/sources. Do not run formal eval, 64-sample
eval, grids, retraining, or scorer/rerank-mode changes from this report alone.

Validation:

```text
py_compile scripts/analyze_expanded_pairing_oracle_absent_drift_diversity.py: passed
git diff --check: passed
git diff --cached --check: passed
```

## Previous Codex Workflow

Codex recorded the user-run expanded-pairing 16-sample residual-debug smoke and
safety parse:

```text
scripts/run_expanded_pairing_residual_debug_smoke16.sh
residual_debug_expanded_pairing_smoke16_report.md
residual_debug_safety_expanded_pairing_smoke16.md
residual_debug_safety_expanded_pairing_smoke16.json
```

No new experiment was run by Codex in this documentation workflow. The user-run
smoke was not a formal 32-sample eval, 64-sample eval, grid, retraining run,
scorer change, candidate-generation change, or rerank mode change.

Summary:

```text
baseline candidates logged=201
baseline oracle=2/16
baseline pair-oracle=1/16
baseline selected=1/16
expanded candidates logged=350
expanded oracle=5/16
expanded pair-oracle=4/16
expanded selected=0/16
parse failures=0
fingerprint failures=0
active residual vector length=18
weak residual vector length=96
offline per_active_dim_norm_plus_weak selected hits=0
offline selected-hit harms=0
offline selected-miss rescues=0/5
offline selected pair candidates=8
```

Oracle ranks under the offline score:

```text
sample 5: rank 8
sample 8: rank 6
sample 11: rank 3
sample 12: rank 3
sample 13: rank 11
```

Interpretation: expanded pairing improves oracle availability but selected
recovery fails, and per-active-dimension normalization rescues none of the
oracle-present misses. The `selected-hit harms=0` count is not strong safety
evidence because expanded-pairing selected hits were already 0. Do not add
`per_active_dim_norm_plus_weak` as a rerank mode and do not run a formal
32-sample eval for this scorer idea. Return to drift span diversity for
expanded-pairing oracle-absent samples, or deeper fingerprint ambiguity
diagnostics.

Validation:

```text
py_compile sde_validation_probe.py scripts/analyze_residual_debug_safety.py: passed
git diff --check: passed
```

## Previous Codex Workflow

Codex ran one 16-sample residual-debug smoke and offline safety parse:

```text
/private/tmp/gensr_sde_residual_debug_smoke16.log
/private/tmp/gensr_sde_residual_debug_smoke16.json
residual_debug_logging_smoke16_report.md
residual_debug_safety_smoke16.md
residual_debug_safety_smoke16.json
```

This was a single smoke plus offline parse. It did not run a formal 32-sample
eval, 64-sample eval, grids, retraining, scorer changes, candidate-generation
changes, or rerank mode changes.

Summary:

```text
samples logged=16
candidates logged=201
valid candidates=201
parse failures=0
fingerprint failures=0
active residual vector length=18
weak residual vector length=96
samples with oracle=2
selected hits=1
pair oracle samples=1
selected-hit harms=0
oracle-present selected-miss rescues=0
```

Interpretation: the smoke is no longer completely inconclusive because it
contains oracle and selected-hit cases. The offline
`per_active_dim_norm_plus_weak` diagnostic preserved the one selected hit and
pair-oracle hit, but did not rescue the one oracle-present miss. No formal eval
or scorer implementation is justified from this result alone. If scorer safety
needs one more check, run a future 16-sample expanded-pairing residual-debug
smoke; otherwise return to drift span diversity or deeper fingerprint ambiguity
diagnostics.

Validation:

```text
py_compile sde_validation_probe.py scripts/analyze_residual_debug_safety.py: passed
16-sample smoke: passed
safety helper: passed
```

## Previous Codex Workflow

Codex added richer residual-vector logging and ran one 8-sample smoke:

```text
sde_validation_probe.py --rerank-residual-debug-json
scripts/analyze_residual_debug_safety.py
residual_debug_logging_smoke_report.md
residual_debug_safety_smoke8.md
residual_debug_safety_smoke8.json
/private/tmp/gensr_sde_residual_debug_smoke8.json
/private/tmp/gensr_sde_residual_debug_smoke8.log
```

This was a code/logging plus small smoke task. It did not run a formal
32-sample eval, 64-sample eval, grids, retraining, scorer changes, or rerank
mode changes.

Summary:

```text
new flag=--rerank-residual-debug-json
samples logged=8
candidates logged=101
valid candidates=101
parse failures=0
fingerprint failures=0
active residual vector length=18
weak residual vector length=96
samples with oracle=0
selected hits=0
pair oracle samples=0
```

The debug JSON contains candidate-level source/rank labels, selected/oracle/pair
labels, role-wise constants, normalized model score, active/weak scores, and
active/weak residual vectors. The safety helper ran
`per_active_dim_norm_plus_weak`, but the smoke had no oracle candidates or
selected hits, so it validates logging and is inconclusive for scorer safety.

Validation:

```text
py_compile sde_validation_probe.py scripts/analyze_residual_debug_safety.py: passed
8-sample smoke: passed
safety helper: passed
```

## Previous Codex Workflow

Codex added an offline expanded-pairing active-residual scorer ablation:

```text
scripts/analyze_expanded_pair_active_ablation.py
expanded_pairing_active_residual_ablation.md
expanded_pairing_active_residual_ablation.json
/private/tmp/gensr_sde_expanded_pair_active_ablation.log
```

This used existing residual/candidate logs only. It did not run
`sde_validation_probe.py`, model decoding, candidate regeneration, formal eval,
64-sample eval, grids, retraining, or scorer/default changes.

Summary:

```text
selected misses analyzed=12
oracle-only-pair misses=7
wrong selected pair cases=8
known selected-hit top2 checks=3
pair-oracle hit checks=1
best offline variant=per_active_dim_norm_plus_weak
best offline variant rescues=6/12
oracle-only-pair rescues=4/7
debug top2 hit harms for best variant=0
decision=C
```

Clipped/Huber/top-k active variants rescued 3-5 misses but harmed one debug
top-2 selected-hit check. Per-active-dimension normalization was strongest, but
it was calibrated only from miss residuals. Recommendation: do not add a rerank
mode and do not run a formal eval; if scorer work continues, add richer
residual logging in a small 8- or 16-sample smoke first.

Validation:

```text
py_compile scripts/analyze_expanded_pair_active_ablation.py: passed
helper run: completed in about 7 seconds
```

## Previous Codex Workflow

Codex added targeted expanded-pairing active-residual trap diagnostics:

```text
scripts/analyze_expanded_pair_active_residuals.py
expanded_pairing_active_residual_trap_diagnostics.md
expanded_pairing_active_residual_trap_diagnostics.json
/private/tmp/gensr_sde_expanded_pair_active_residuals.log
```

The helper did not run model decoding, candidate regeneration, formal eval,
64-sample eval, retraining, active/weak grids, or scorer/default changes. It
recomputed residual vectors only for the 12 existing expanded-pairing selected
misses.

Summary:

```text
selected misses analyzed=12
wrong selected pair cases=8
oracle-only-pair misses=7
active favors selected=12/12
weak favors oracle while active favors selected=4/12
outlier-dominated active traps=7/12
broad active traps=2/12
mixed active traps=3/12
KM1/drift-leaning active advantages=10/12
```

Interpretation: the active trap is often caused by a few active residual
dimensions, mostly KM1/drift-like. This supports one future offline
robust/clipped active-residual ablation on existing expanded-pairing candidates
only. It does not justify a formal eval, new rerank mode, pair pruning, or
expanded-pairing default.

Validation:

```text
py_compile scripts/analyze_expanded_pair_active_residuals.py: passed
helper run: completed in about 16 seconds
```

## Previous Codex Workflow

Codex added a log/JSON-only wrong-pair pruning diagnostic:

```text
scripts/analyze_expanded_pair_pruning.py
expanded_pairing_wrong_pair_pruning_diagnostics.md
expanded_pairing_wrong_pair_pruning_diagnostics.json
/private/tmp/gensr_sde_expanded_pair_pruning.log
```

No eval, candidate regeneration, formal rerank run, 64-sample run, retraining,
or scorer change was launched.

Summary:

```text
selected misses analyzed=12
wrong selected pair cases=8
pair-oracle samples to preserve=8
pair-oracle source ranks=2..15
approx wrong-pair source ranks=1..14
active-distance traps among wrong selected pairs=5/8
safe_pruning_rule_found=0
```

Interpretation: no safe oracle-preserving pruning criterion is visible from the
existing logs/JSON. Pair source-rank pruning would remove real pair oracles
unless the threshold is at least 15, which also keeps observed wrong pair
selections. Structural pruning is unsafe because harmful mean-reverting or
constant-drift pair candidates overlap with true pair-oracle families.

Recommendation: do not add pair pruning or a blanket pair bonus. Next perform
targeted residual analysis of `active_kramers_moyal` traps for wrong pair
candidates versus pair oracles.

Validation:

```text
py_compile scripts/analyze_expanded_pair_pruning.py: passed
git diff --check: passed
```

## Previous Codex Workflow

Codex added a log-only expanded-pairing oracle-miss ranking diagnostic:

```text
scripts/analyze_expanded_pairing_misses.py
expanded_pairing_oracle_miss_ranking_diagnostics.md
expanded_pairing_oracle_miss_ranking_diagnostics.json
/private/tmp/gensr_sde_expanded_pairing_miss_ranking.log
```

No eval, candidate regeneration, retraining, scorer change, or 64-sample run
was launched.

Summary over the 12 expanded-pairing selected misses:

```text
oracle-only-pair misses=7
beam/sampling oracle score misses=5
same diffusion but wrong drift=4
same drift but wrong diffusion=4
both sides wrong=4
selected source is pair=8
selected source is beam=4
near score gaps <=0.10=2/12
oracle-only-pair near score gaps <=0.10=1/7
```

Main interpretation: expanded pairing's new oracle candidates usually lose on
the active+weak role-wise score, not by tiny tie-break margins. For all 7
oracle-only-pair misses, the selected non-oracle's advantage is dominated by
`active_kramers_moyal`. Model score conflict is not systematic for pair-only
misses (`1/7`), though it appears in most beam/sampling score misses (`4/5`).

Recommendation: do not add a blanket pair bonus or simple pair-aware tie-break
from this log alone. Next inspect high-ranking wrong pair candidates and
candidate pruning criteria using existing logs.

Validation:

```text
py_compile scripts/analyze_expanded_pairing_misses.py: passed
git diff --check: passed
```

## Previous Codex Workflow

Codex parsed the user-run expanded-pairing formal 32-sample eval log:

```text
/private/tmp/gensr_sde_32_expanded_pairing_rolewise.log
expanded_pairing_32_eval_report.md
```

No new experiment was launched. Key formal eval metrics:

```text
selected exact/relaxed=3/32
oracle exact/relaxed=15/32
pair oracle exact/relaxed=8/32
valid candidates=715
unique candidate avg=22.343750
unique paired candidate avg=12.906250
```

Comparison to the previous strongest baseline:

```text
baseline selected exact/relaxed=5/32
baseline oracle exact/relaxed=9/32
baseline pair oracle exact/relaxed=2/32
```

Interpretation: expanded pairing successfully raises oracle coverage in formal
eval, but selected recovery regresses. It should not become the default and no
64-sample expansion is justified yet. The next task is to analyze
expanded-pairing oracle-miss ranking from existing logs, especially the 7
oracle-only-pair misses.

Validation:

```text
py_compile sde_validation_probe.py: passed
git diff --check: passed
```

## Previous Codex Workflow

Codex ran an expanded-pairing candidate coverage probe:

```text
candidate_coverage_pair_expanded.md
candidate_coverage_pair_expanded.json
pairing_expansion_coverage_report.md
/private/tmp/gensr_sde_candidate_coverage_pair_expanded.log
```

Expanded pairing configuration:

```text
pair_drift_topk=5
pair_diffusion_topk=6
pair_drift_diffusion_candidates=32
```

This was candidate-generation coverage only: no formal rerank eval, no
64-sample eval, no retraining, no active/weak grid, and no scorer/default
change.

Result:

```text
baseline full_oracle_present=9/32
expanded full_oracle_present=15/32
baseline oracle_absent=23/32
expanded oracle_absent=17/32
```

All six pairing-missing samples were rescued as pair-source full oracle
candidates:

```text
5, 8, 11, 23, 30, 31
```

Interpretation: `pair_drift_diffusion_candidates=32` was enough for the five
cap-blocked cases; `pair_drift_topk=5` was enough for sample 23, which also
needed the larger cap because its combined pair rank was 24. One future formal
32-sample eval with the current strongest scorer and expanded pairing is now
justified. Remaining oracle-absent samples are all drift-missing.

Validation:

```text
py_compile scripts/analyze_candidate_coverage.py scripts/analyze_drift_pairing_coverage.py: passed
git diff --check: passed
```

## Previous Codex Workflow

Codex restored the checkpoint-backed candidate coverage path and completed the
full drift/pairing rank diagnostic:

```text
candidate_coverage_diagnostics.json
drift_pairing_coverage_diagnostics.md
/private/tmp/gensr_sde_candidate_coverage_json.log
/private/tmp/gensr_sde_drift_pairing_coverage_full.log
```

Checkpoint status:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth: present, real file
checkpoints/gensr_sde_role_token_2000/role_token_2000.pth: present backup
checkpoints/: ignored by git
```

Candidate coverage JSON generation passed and reproduced:

```text
full_oracle_present=9/32
oracle_absent_missing_sides=drift:17,pairing:6
```

Full pairing rank diagnostic:

```text
drift_missing=17
pairing_missing=6
top_missing_drift_family=linear drift:6
pairing_actions=increase_pair_candidate_cap:5,increase_pair_drift_topk:1
```

Pairing-missing samples `5`, `8`, `11`, `30`, and `31` have exact drift and
diffusion within current top-k but are blocked by pair candidate cap. Sample
`23` needs the exact drift admitted by raising `pair_drift_topk` to at least
5; its combined pair rank would be 24 after top-k expansion.

Validation:

```text
py_compile scripts/analyze_candidate_coverage.py scripts/analyze_drift_pairing_coverage.py: passed
git diff --check: passed
```

## Previous Codex Workflow

Codex added drift/pairing coverage diagnostics:

```text
scripts/analyze_drift_pairing_coverage.py
drift_pairing_coverage_diagnostics.md
/private/tmp/gensr_sde_drift_pairing_coverage.log
```

`scripts/analyze_candidate_coverage.py` was also extended to emit a JSON sidecar
with span ranks for future pairing analysis.

The exact rank diagnostic is currently blocked because:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

is missing. The helper therefore wrote a partial report from the existing
candidate coverage markdown.

Available drift-missing taxonomy:

```text
diffusion_only_present samples = 17
linear drift missing = 6
sin drift missing = 6
nested-mul linear drift missing = 3
polynomial-like drift missing = 2
```

Pairing-missing samples:

```text
5, 8, 11, 23, 30, 31
```

Exact drift/diffusion ranks, top-k membership, pair-cap checks, and
dedup/filtering diagnosis require regenerating `candidate_coverage_diagnostics.json`
after restoring/regenerating the checkpoint.

Validation:

```text
py_compile scripts/analyze_candidate_coverage.py scripts/analyze_drift_pairing_coverage.py: passed
drift/pairing helper partial run: passed
git diff --check: passed
```

## Previous Codex Workflow

Codex added candidate coverage diagnostics:

```text
scripts/analyze_candidate_coverage.py
candidate_coverage_diagnostics.md
/private/tmp/gensr_sde_candidate_coverage.log
```

This was not a formal eval. The helper loaded the existing 2000-step checkpoint
read-only and reran candidate generation for the 32 held-out samples without
fingerprint scoring, retraining, 64-sample expansion, active/weak grids, or a
new rerank heuristic.

Coverage summary:

```text
full_oracle_present=9/32
scorer_selected_oracle=5/32
oracle_present_but_selected_missed=4/32
drift_only_present=0
diffusion_only_present=17
drift_and_diffusion_separately_present_but_not_paired=6
neither_side_present=0
```

Source breakdown:

```text
beam:     full oracle 6, exact drift 15, exact diffusion 32
sampling: full oracle 1, exact drift 3,  exact diffusion 8
pair:     full oracle 2, exact drift 7,  exact diffusion 17
```

Interpretation: the 23 oracle-absent samples are not missing diffusion. The
main blocker is drift span generation: 17/23 oracle-absent samples already have
the truth diffusion but not the truth drift. The other 6/23 have exact drift and
exact diffusion separately but fail to pair them into a full oracle. Next work
should prioritize drift candidate diversity, then pairing coverage/ranking.

Validation:

```text
py_compile scripts/analyze_candidate_coverage.py: passed
candidate coverage diagnostic run: passed
git diff --check: passed
```

## Earlier Codex Workflow

Codex added offline residual score ablation:

```text
scripts/analyze_residual_score_ablation.py
rolewise_residual_score_ablation.md
```

Because the prior residual-vector log was not machine-readable enough, Codex
extended `scripts/analyze_rolewise_residuals.py` to emit:

```text
rolewise_residual_vector_diagnostics.json
```

This was offline analysis only: no formal 32-sample eval, no 64-sample eval, no
retraining, no active/weak grid, and no new rerank mode/default change.

Results over samples `13,19,24,26`:

```text
current active+weak L2: 0/4 oracle wins
L1 residual sum: 0/4 oracle wins
L2 residual sum: 0/4 oracle wins
mild clipped/Huber/per-segment normalization: 0/4 oracle wins
aggressive clipping/top-k removal: 1/4 oracle wins
```

Only sample 19 flipped, and only under aggressive `clipped_l2_p90`. Sample 24
(drift-side ambiguity) and sample 26 (diffusion-side ambiguity) did not flip
under any tested offline variant.

Interpretation: a formal 32-sample robust-scoring eval is not justified yet.
The remaining misses look more like fingerprint ambiguity / insufficient
candidate evidence than an easy feature-normalization scorer fix.

Validation:

```text
py_compile scripts/analyze_rolewise_residuals.py scripts/analyze_residual_score_ablation.py: passed
targeted residual helper rerun for JSON sidecar: passed
offline score ablation: passed
git diff --check: passed
```

## Earlier Codex Workflow

Codex added a narrow helper script:

```text
scripts/analyze_rolewise_residuals.py
```

and generated:

```text
rolewise_residual_vector_diagnostics.md
/private/tmp/gensr_sde_rolewise_residual_vectors.log
```

This was not a formal 32-sample eval. The helper reproduced only the known
remaining role-wise miss samples `13,19,24,26` from the eval seed and scored
selected/oracle role-swap combinations. No retraining, 64-sample eval,
active/weak grid, target-format change, fingerprint-schema change, checkpoint
overwrite, or data-pickle overwrite was performed.

Residual-vector classifications:

```text
sample 13: both-side ambiguity; feature-scaling artifact; model-score conflict
sample 19: both-side ambiguity; feature-scaling artifact
sample 24: drift-side ambiguity; feature-scaling artifact
sample 26: diffusion-side ambiguity; feature-scaling artifact
```

Key interpretation: the remaining four role-wise misses still favor selected
non-oracles on aggregate active+weak residuals. Samples 24 and 26 isolate the
problem cleanly because one role is shared: sample 24 is drift-side ambiguity,
sample 26 is diffusion-side ambiguity. Do not add a new rerank heuristic from
this alone; the next low-cost move is targeted feature normalization /
residual-scaling diagnostics, then candidate generation if the scorer remains
ambiguous.

Validation:

```text
py_compile scripts/analyze_rolewise_residuals.py: passed
git diff --check: passed
```

## Earlier Codex Workflow

Codex added role-wise constant-grid scoring to `sde_validation_probe.py` and ran
the requested 32-sample eval-only calibration experiments. No training was run,
no 64-sample probe was run, and no checkpoint or dataset pickle was overwritten.

New score mode:

```text
constant_grid_rolewise_no_multi_u0
```

It keeps the active+weak no-`multi_u0` score, but searches separate shared
constants for the drift and diffusion roles over `--rerank-constant-values`.
The probe also prints role-wise best drift/diffusion constants and shared-vs-
rolewise rescue summaries.

32-sample results:

```text
shared constant baseline:
  selected exact/relaxed=4/32
  oracle exact/relaxed=9/32
  pair oracle exact/relaxed=2/32

role-wise constant, active:weak 1:1:
  selected exact/relaxed=5/32
  oracle exact/relaxed=9/32
  pair oracle exact/relaxed=2/32
  rescued shared misses=1/5

role-wise constant, active:weak 2:1:
  selected exact/relaxed=3/32
  oracle exact/relaxed=9/32
  pair oracle exact/relaxed=2/32
  rescued shared misses=0/6

role-wise constant, active:weak 1:0.5:
  selected exact/relaxed=3/32
  oracle exact/relaxed=9/32
  pair oracle exact/relaxed=2/32
  rescued shared misses=0/6
```

Main rescue:

```text
sample_index=17
shared selected diffusion: mul CONSTANT abs x_0
oracle / rolewise selected diffusion: mul CONSTANT sqrt abs x_0
rolewise constants: drift=1, diffusion=0.5
rolewise selected/oracle score=0.818141
shared selected/oracle scores=0.872129 / 0.903798
```

Interpretation: role-wise constants are worth keeping as the current strongest
setting, but simple active/weak reweighting is brittle. The remaining misses are
not solved by constant handling alone: even when role-wise constants lower the
oracle score, non-oracles still often have lower active+weak distance.

Raw logs:

```text
/private/tmp/gensr_sde_rolewise_smoke_8.log
/private/tmp/gensr_sde_32_rolewise_no_multi_u0.log
/private/tmp/gensr_sde_32_rolewise_active2_weak1.log
/private/tmp/gensr_sde_32_rolewise_active1_weak05.log
```

## Earlier Codex Workflow

Codex added oracle gap diagnostics to `sde_validation_probe.py` and reran the
32-sample no-tie eval-only checkpoint probe. No training was run, no target
format changed, and no checkpoint or dataset pickle was overwritten.

New diagnostic output:

- `oracle_case_summary`
- `oracle_miss_type_summary`
- `oracle_miss_cases`

The per-miss output includes sample index, truth, selected sequence, best oracle
sequence, selected/oracle sources, selected/oracle rerank score, score gap,
best constants, normalized model scores, drift/diffusion tokens, segment
distances, shared-drift/shared-diffusion flags, miss side, and whether the
oracle appeared only from pairing.

Validation:

```text
py_compile sde_validation_probe.py: passed
git diff --check: passed
8-sample eval-only output smoke: passed
32-sample eval-only oracle-gap diagnostic: passed
```

32-sample oracle-gap summary:

```text
total_samples=32
samples_with_any_oracle_candidate=9
samples_where_selected_is_oracle=4
samples_where_oracle_exists_but_selected_misses=5
oracle_best_source_counts beam=6 sampling=1 pair=2 unknown=0
oracle_any_source_counts beam=6 sampling=1 pair=2 unknown=0
selected_hit_source_counts beam=3 sampling=0 pair=1 unknown=0
```

Miss type summary:

```text
constant_mismatch=0
same_diffusion_but_wrong_drift=2
same_drift_but_wrong_diffusion=1
both_drift_and_diffusion_wrong=2
oracle_lower_model_score_candidate=2
oracle_only_appears_from_pair=1
oracle_from_beam_or_sampling_but_score_misses=4
parse_failures=0
fingerprint_failures=0
```

Key interpretation: the immediate `4/32` selected versus `9/32` oracle gap is
mostly rerank score / constant-handling behavior. In all 5 miss cases, an oracle
template is present, but the active/weak score selects a non-oracle. Candidate
generation remains the larger ceiling because 23/32 samples still have no oracle
candidate.

Raw logs:

```text
/private/tmp/gensr_sde_oracle_diag_smoke_8.log
/private/tmp/gensr_sde_32_oracle_gap_diagnostics.log
```

## Previous Codex Workflow

Codex validated the near-tie rerank tie-breaks on a larger 32-sample eval-only
held-out probe. No training was run, and no checkpoint or dataset pickle was
overwritten.

Checkpoint:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

Shared eval settings:

```text
eval_samples=32
batch_size=8
n_paths=800
active_paths=800
n_steps=60
constrained_beam_size=8
rerank_score=constant_grid_componentwise_no_multi_u0
rerank_constant_values=0.25,0.5,1.0,2.0,4.0
rerank_tie_epsilon=0.005
sample_candidates=8
sample_temperatures=0.8,1.0,1.2
pair_drift_diffusion_candidates=8
```

32-sample comparison:

```text
baseline no tie:
  greedy exact/relaxed=0/32
  constrained beam exact/relaxed=0/32
  selected reranked exact/relaxed=4/32
  oracle exact/relaxed=9/32
  pair oracle exact/relaxed=2/32

active_distance tie-break:
  selected reranked exact/relaxed=4/32
  oracle exact/relaxed=9/32
  pair oracle exact/relaxed=2/32

state_dependent_drift tie-break:
  selected reranked exact/relaxed=4/32
  oracle exact/relaxed=9/32
  pair oracle exact/relaxed=2/32
```

Candidate diagnostics were identical across the three settings:

```text
rerank_candidates_attempted=405
rerank_candidates_valid=405
rerank_fingerprint_failures=0
rerank_unique_candidate_avg=12.656250
rerank_unique_drift_avg=4.468750
rerank_unique_diffusion_avg=5.000000
rerank_unique_paired_candidate_avg=3.218750
```

Debug notes:

- Neither `active_distance` nor `state_dependent_drift` improved selected
  exact/relaxed over no-tie baseline on 32 samples.
- Both tie-breaks stayed below the oracle ceiling (`4/32` selected versus
  `9/32` oracle).
- The printed debug top-k did not show multi-candidate tie groups
  (`tie_group_rank=2` did not appear), so no clear tie-break-specific mistaken
  selection was visible in the logged debug cases.
- Because neither tie-break exceeded baseline and each 32-sample constant-grid
  + pairing run was CPU-expensive, the 64-sample extension was not run.

Raw logs for local follow-up:

```text
/private/tmp/gensr_sde_32_baseline_no_tie.log
/private/tmp/gensr_sde_32_active_distance.log
/private/tmp/gensr_sde_32_state_dependent_drift.log
```

Interpretation: the 16-sample near-tie hit was useful diagnostically, but it is
not yet stable enough to make either tie-break the default. The next priority is
closing the selected-versus-oracle gap through candidate generation, pairing
coverage, and constant handling.

## Previous Codex Workflow

Codex added epsilon tie-breaking for near-equal rerank distances in
`sde_validation_probe.py`.

Key changes:

- added `--rerank-tie-epsilon`;
- added `--rerank-tie-break none|model_score|active_distance|state_dependent_drift`;
- added `--rerank-score constant_grid_active_weak`, which uses
  `--rerank-component-weights` for active/weak score variants such as
  `0.0,2.5,1.0`;
- debug output now prints tie group candidates, the tie-break mode, the selected
  tie candidate, active/weak distances, and `state_dependent_drift`.

Validation:

- `py_compile sde_validation_probe.py`: passed.
- `git diff --check`: passed.
- Small 2-step smoke with tie-break diagnostics: passed.
- Eval-only checkpoint probe with
  `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth`: passed on 16
  held-out samples for four settings: no tie-break baseline,
  `active_distance` tie-break, `state_dependent_drift` tie-break, and
  active-heavy weights.

16-sample checkpoint probe result for successful tie-break settings:

```text
rerank_candidates_attempted=167
rerank_candidates_valid=167
rerank_fingerprint_failures=0
rerank_pair_candidates_attempted=56
rerank_pair_candidates_valid=56
reranked_sequence_exact=0.062500
reranked_sequence_relaxed_no_constants=0.062500
rerank_oracle_sequence_exact=0.062500
rerank_oracle_sequence_relaxed_no_constants=0.062500
rerank_pair_oracle_sequence_exact=0.062500
rerank_pair_oracle_sequence_relaxed_no_constants=0.062500
rerank_unique_candidate_avg=10.437500
rerank_unique_drift_avg=3.000000
rerank_unique_diffusion_avg=5.000000
rerank_unique_paired_candidate_avg=3.500000
```

Tie-break comparison:

```text
baseline no tie:
  selected exact/relaxed=0/0
  inspected oracle_rank=2
active_distance tie-break:
  selected exact/relaxed=1/16
  inspected oracle_rank=1
state_dependent_drift tie-break:
  selected exact/relaxed=1/16
  inspected oracle_rank=1
active-heavy weights 0.0,2.5,1.0:
  selected exact/relaxed=0/0
  inspected oracle_rank=2
```

Inspected tie group:

```text
oracle distance=1.507198
top non-oracle distance=1.505882
epsilon=0.005
oracle active=0.875270
top non-oracle active=0.961477
oracle state_dependent_drift=1
top non-oracle state_dependent_drift=0
```

Interpretation: the near-tie diagnosis was actionable. Both `active_distance`
and `state_dependent_drift` tie-breaks select the paired oracle in the inspected
case and produce the first selected reranked exact/relaxed hit. The next step is
checking whether this survives 32/64-sample probes and does not introduce new
false positives.

## Previous Codex Workflow

Codex updated `AGENTS.md` with a new Web GPT Snapshot Rule:

- after any Codex workflow that changes project state, update
  `web_gpt_snapshot/02_RECENT_CHANGES.md` and
  `web_gpt_snapshot/03_CURRENT_TASK.md`;
- update `01_PROJECT_STATUS.md`, `04_COMMANDS.md`, and `00_WEB_GPT_INDEX.md`
  only when their respective content actually changes;
- keep fixed small snapshot files stable for manual replacement in ChatGPT Web
  Project Sources;
- do not copy datasets, checkpoints, dumps, weights, `.git/`, caches, or full
  `symbolicregression/` into the snapshot.

This update also refreshed this file and `03_CURRENT_TASK.md`. No Python source,
commands, checkpoints, or experiment metrics changed.

## Previous Codex Workflow

Codex cleaned the git chain and committed the SDE role-token validation state:

```text
4affb88 Add role-token SDE validation probe
```

That commit added/updated:

- `sde_validation_probe.py`: fast SDE train/eval harness with checkpoint save/load/eval-only and constrained beam.
- `sde_dataset_generator.py`: role-token target format and split drift/diffusion encodings.
- `symbolicregression/envs/environment.py`: added `<DRIFT>` and `<DIFFUSION>` tokens.
- `symbolicregression/trainer_vae.py`: optional split SDE loss path and CPU-safe `to_cuda`.
- `train.py`: corrected SDE sample injection through `EnvDataset.generate_sample`.
- `README.md`, `PROJECT_STATUS.md`, `SDE_TRAINING_REPORT.md`, `AGENTS.md`: recovery docs and experiment state.

Codex then generated an older, fuller handoff directory:

```text
web_gpt_handoff/
```

The current request creates this lighter long-term snapshot:

```text
web_gpt_snapshot/
```

## Git Status Before This Rule Update

```text
?? web_gpt_handoff/
?? web_gpt_snapshot/
```

There was no tracked source diff before updating `AGENTS.md` and the snapshot
notes.

## Recent Diff Summary

- Tracked documentation change: `AGENTS.md` now includes the Web GPT Snapshot
  Rule.
- Snapshot notes updated in `web_gpt_snapshot/02_RECENT_CHANGES.md` and
  `web_gpt_snapshot/03_CURRENT_TASK.md`.
- No datasets, checkpoints, weights, dumps, or cache files were added to this snapshot.

## Verification Already Run In Recent Workflow

Passed:

- `py_compile` on edited Python entrypoints.
- `git diff --check`.
- `git diff --cached --check` before the commit.
- 2-step `sde_validation_probe.py` smoke test with checkpoint save and held-out evaluation.

Not rerun during this documentation-only update:

- Full eval-only checkpoint decoding.
- 2000-step retraining.
- Python smoke tests, because no Python source changed.

## Risks / Unfinished Items

- The best checkpoint lives in `/private/tmp` and may be deleted.
- First-pass reranking is implemented but has not improved exact/relaxed recovery yet.
- Pairing produced nonzero oracle hits, but selected reranked exact/relaxed
  metrics remained zero in a 16-sample checkpoint eval.
- Componentwise scoring did not select the oracle candidate either; score
  diagnostics point to segment distances penalizing the oracle template.
- Future project-state changes must update `02_RECENT_CHANGES.md` and
  `03_CURRENT_TASK.md`; update other fixed snapshot files only when their
  contents actually change.
