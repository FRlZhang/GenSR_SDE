# Recent Changes

## Last Codex Workflow

Codex added rerank score diagnostics and componentwise scoring for paired
candidates in `sde_validation_probe.py`.

Key changes:

- added `--rerank-score componentwise`;
- added `--rerank-component-weights`, defaulting to `1.0,2.0,1.0`;
- debug output now includes full fingerprint distance and segment distances for
  `multi_u0_moments`, `active_kramers_moyal`, and `gaussian_weak_kernel`;
- debug output now reports oracle candidate rank, oracle distance, top-ranked
  distance, and oracle-minus-top distance delta when an oracle candidate exists.

Validation:

- `py_compile sde_validation_probe.py`: passed.
- `git diff --check`: passed.
- Small 2-step smoke with componentwise reranking diagnostics: passed.
- Eval-only checkpoint probe with
  `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth`: passed on 16
  held-out samples for both `full_fingerprint` and `componentwise`.

16-sample checkpoint probe result:

```text
rerank_candidates_attempted=167
rerank_candidates_valid=167
rerank_fingerprint_failures=0
rerank_pair_candidates_attempted=56
rerank_pair_candidates_valid=56
reranked_sequence_exact=0.000000
reranked_sequence_relaxed_no_constants=0.000000
rerank_oracle_sequence_exact=0.062500
rerank_oracle_sequence_relaxed_no_constants=0.062500
rerank_pair_oracle_sequence_exact=0.062500
rerank_pair_oracle_sequence_relaxed_no_constants=0.062500
rerank_unique_candidate_avg=10.437500
rerank_unique_drift_avg=3.000000
rerank_unique_diffusion_avg=5.000000
rerank_unique_paired_candidate_avg=3.500000
```

Full and componentwise had the same selected exact/relaxed outcome: selected
reranked exact/relaxed stayed 0 while oracle stayed 1/16.

Observed full-fingerprint oracle diagnostic:

```text
oracle_rank=11
top_ranked_distance=1.122157
oracle_distance=1.392796
oracle_top_distance_delta=0.270639
oracle multi_u0_moments=2.564722
top multi_u0_moments=1.704655
oracle active_kramers_moyal=1.001000
top active_kramers_moyal=1.278720
oracle gaussian_weak_kernel=1.353263
top gaussian_weak_kernel=0.931437
```

Observed componentwise oracle diagnostic:

```text
oracle_rank=8
top_ranked_distance=5.126071
oracle_distance=5.919984
oracle_top_distance_delta=0.793913
oracle multi_u0_moments=2.564722
top multi_u0_moments=2.131954
oracle active_kramers_moyal=1.001000
top active_kramers_moyal=0.926156
oracle gaussian_weak_kernel=1.353263
top gaussian_weak_kernel=1.141805
```

Interpretation: componentwise improved the inspected oracle candidate from rank
11 to rank 8, but still did not select it. Under componentwise scoring, every
segment was worse for the oracle than for the top-ranked candidate, with the
largest gap in `multi_u0_moments`. This suggests the next issue may be constant
sensitivity from evaluating all `CONSTANT` tokens as `1.0`, rather than
fingerprint redesign.

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
