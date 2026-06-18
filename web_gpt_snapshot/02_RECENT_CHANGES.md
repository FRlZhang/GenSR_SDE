# Recent Changes

## Last Codex Workflow

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
