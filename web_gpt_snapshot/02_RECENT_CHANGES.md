# Recent Changes

## Last Codex Workflow

Codex added top non-oracle template diagnostics and extra constant-grid score
variants in `sde_validation_probe.py`.

Key changes:

- added `--rerank-score constant_grid_componentwise_no_multi_u0`;
- added `--rerank-score constant_grid_active_only`;
- added `--rerank-score constant_grid_moments_downweighted`;
- when an oracle candidate exists, debug output now prints candidates ranked
  before it, their best constants, paired/non-paired status, model scores,
  segment distances, raw segment deltas, weighted segment deltas, and the
  dominant score-gap segment.

Validation:

- `py_compile sde_validation_probe.py`: passed.
- `git diff --check`: passed.
- Small 2-step smoke with top non-oracle diagnostics: passed.
- Eval-only checkpoint probe with
  `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth`: passed on 16
  held-out samples for four score variants:
  `constant_grid_componentwise`,
  `constant_grid_componentwise_no_multi_u0`, `constant_grid_active_only`, and
  `constant_grid_moments_downweighted`.

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

Score variant comparison on the inspected oracle case:

```text
constant_grid_componentwise:
  oracle_rank=5, oracle_distance=5.125097, top_distance=3.722727
constant_grid_componentwise_no_multi_u0:
  oracle_rank=2, oracle_distance=1.507198, top_distance=1.505882
constant_grid_active_only:
  oracle_rank=4, oracle_distance=0.875270, top_distance=0.754370
constant_grid_moments_downweighted:
  oracle_rank=11, oracle_distance=3.996443, top_distance=2.803234
```

All variants kept selected reranked exact/relaxed at 0 while oracle exact/relaxed
stayed 1/16.

Closest non-oracle gap under `constant_grid_componentwise_no_multi_u0`:

```text
oracle:
  source=pair
  best_constant=4
  drift=mul mul CONSTANT CONSTANT sin x_0
  diffusion=add CONSTANT mul CONSTANT abs x_0
top non-oracle:
  source=beam
  best_constant=1
  drift=mul mul CONSTANT CONSTANT CONSTANT
  diffusion=add CONSTANT mul CONSTANT abs x_0
distance_delta=0.001317
weighted active delta=-0.086207
weighted weak delta=0.087524
dominant score-gap segment=gaussian_weak_kernel
```

Interpretation: dropping `multi_u0_moments` nearly selects the oracle. The
remaining miss is a tiny active+weak tradeoff: the oracle is better on active
moments, but the non-oracle wins slightly more on weak-kernel distance. The next
step is lightweight tie-breaking or structure-aware penalties, not fingerprint
redesign or retraining.

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
