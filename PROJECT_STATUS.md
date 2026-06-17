# PROJECT_STATUS.md

## Current Goal

Improve sequence-level SDE recovery after establishing that the current unified
fingerprint and role-token target are learnable by GenSR. Candidate generation
plus fingerprint-distance reranking is now implemented in the validation probe;
the next step is improving candidate diversity and rerank scoring because the
first small eval did not improve exact/relaxed recovery.

## Last Updated

2026-06-17

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
- Logged validated experiments in
  [SDE_TRAINING_REPORT.md](/Users/lzhang/Documents/GenSR_SDE/SDE_TRAINING_REPORT.md).

## In Progress

- Transitioning from "can the model learn the fingerprint?" to "can we decode
  the right symbolic sequence?".
- Diagnosing why beam+stochastic sampling still misses exact/relaxed target
  templates even after increasing drift/diffusion diversity.

## Next Steps

1. Add stronger structure-level diversity, such as drift/diffusion separate
   generation and pairing, because beam+sampling still has zero oracle hits.
2. Add candidate-pool diagnostics across larger held-out samples and inspect top
   candidates / distances.
3. Compare full-fingerprint, short-moment, and componentwise reranking scores
   after the candidate pool contains exact/relaxed alternatives.
4. Reuse the saved 2000-step checkpoint for decoding experiments instead of
   retraining.
5. Compare greedy, constrained beam, reranked, and oracle candidate metrics on the same
   held-out evaluation setup.
6. Only revisit fingerprint design after candidate diversity and scoring have
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
