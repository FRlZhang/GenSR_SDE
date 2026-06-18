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
tested so far regressed.

## Last Updated

2026-06-18

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
- Logged validated experiments in
  [SDE_TRAINING_REPORT.md](/Users/lzhang/Documents/GenSR_SDE/SDE_TRAINING_REPORT.md).

## In Progress

- Transitioning from "can the model learn the fingerprint?" to "can we decode
  the right symbolic sequence?".
- Using the 32-sample oracle gap diagnostics to choose the next decoding fix:
  rerank scoring and constant handling for the 5 selected misses, then candidate
  generation for the 23 samples without any oracle candidate.

## Next Steps

1. Do not make `active_distance` or `state_dependent_drift` the default based
   on the 16-sample hit; on 32 samples they matched, but did not beat, no tie.
2. Keep `constant_grid_rolewise_no_multi_u0` as the strongest current
   no-retraining decoding setting (`5/32` selected, `9/32` oracle), but do not
   continue broad active/weak weight search because the two tested variants
   regressed.
3. Preserve and inspect drift/diffusion pairing, but treat it as coverage
   support rather than the main selector fix: only 2/9 best oracles came from
   pairs and one miss was pair-only.
4. After the selected/oracle gap is reduced, improve candidate generation for
   the 23/32 samples where no oracle candidate exists.
5. Keep drift/diffusion pairing enabled and tune candidate pool size carefully
   because fingerprint recomputation is CPU-expensive.
6. Reuse the saved 2000-step checkpoint for decoding experiments instead of
   retraining.
7. Compare greedy, constrained beam, reranked, and oracle candidate metrics on the same
   held-out evaluation setup.
8. Only revisit fingerprint design after candidate diversity and scoring have
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
- Candidate generation is still a ceiling: only 9/32 samples had any oracle
  candidate in the current beam+sampling+pair pool.
- The 32-sample constant-grid + pairing probe is CPU-expensive, so 64-sample
  expansion should be reserved for settings that first improve the 32-sample
  selected metrics.
- The main checkpoint currently lives outside the repo in `/private/tmp`, so it
  may disappear.
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
