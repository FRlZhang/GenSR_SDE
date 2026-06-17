# Recent Changes

## Last Codex Workflow

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
