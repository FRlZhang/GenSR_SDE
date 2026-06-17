# Project Status

## Current Goal

Improve sequence-level SDE recovery after establishing that the current unified fingerprint and role-token target are learnable by GenSR. Candidate generation plus fingerprint-distance reranking is now implemented; the next step is improving candidate diversity and rerank scoring because the first small eval did not improve exact/relaxed recovery.

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

## In Progress

- Transitioning from token recognition to full symbolic sequence recovery.
- Diagnosing candidate-pool diversity after oracle metrics showed the current constrained-beam n-best pool did not contain target templates.

## Next Steps

1. Improve candidate diversity beyond the current constrained-beam n-best list; oracle metrics show reranking cannot recover targets that are absent.
2. Inspect rerank debug output across larger held-out samples.
3. Compare full-fingerprint, short-moment, and componentwise scores after candidate diversity improves.
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
- The best checkpoint is outside the repo and may disappear.
- `environment.yml` is upstream/Linux-oriented; recent work used the local macOS `gensr` conda env.

## Best Checkpoint

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

## How To Resume

Read `00_WEB_GPT_INDEX.md`, this file, `02_RECENT_CHANGES.md`, `03_CURRENT_TASK.md`, and `04_COMMANDS.md`. If writing code, temporarily upload only the source files listed in `NEEDED_SOURCE_FILES.md` or in the index.
