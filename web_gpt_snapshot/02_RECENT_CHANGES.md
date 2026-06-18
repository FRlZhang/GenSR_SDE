# Recent Changes

## Last Codex Workflow

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

## Previous Codex Workflow

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
