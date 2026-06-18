# Current Task

## Next Engineering Task

Use the targeted residual-vector diagnostics to decide whether the remaining
role-wise constant misses should be attacked through feature normalization /
score calibration, or whether the scorer is too fingerprint-ambiguous and the
next work should return to candidate generation.

Current strongest no-retraining decoding setting:

```text
rerank_score=constant_grid_rolewise_no_multi_u0
active:weak=1:1
selected exact/relaxed=5/32
oracle exact/relaxed=9/32
pair oracle exact/relaxed=2/32
```

Role-wise constants rescued one shared-constant miss, sample 17, by allowing
drift constant `1.0` and diffusion constant `0.5`. The active-heavy `2:1` and
weak-downweighted `1:0.5` variants both regressed to `3/32`, so do not continue
broad active/weak grid search.

Targeted residual-vector diagnostics now exist:

```text
scripts/analyze_rolewise_residuals.py
rolewise_residual_vector_diagnostics.md
/private/tmp/gensr_sde_rolewise_residual_vectors.log
```

Residual-vector classifications:

```text
sample 13: both-side ambiguity; feature-scaling artifact; model-score conflict
sample 19: both-side ambiguity; feature-scaling artifact
sample 24: drift-side ambiguity; feature-scaling artifact
sample 26: diffusion-side ambiguity; feature-scaling artifact
```

## Primary File To Modify

No primary source file needs to change unless the next task explicitly adds a
new diagnostic. If adding diagnostics, prefer a narrow helper under `scripts/`
before modifying `sde_validation_probe.py`.

Likely next additions:

- inspect whether a few normalized active/weak features dominate the wrong
  selections;
- test feature normalization / residual scaling as diagnostics before any new
  rerank heuristic;
- keep role-wise constant diagnostics in every rerank experiment;
- consider candidate-generation changes only after deciding whether the current
  scorer can reliably choose among existing oracle candidates.

## Supporting Files If Needed

- `sde_fingerprint.py`: read score component semantics only; do not redesign the
  schema first.
- `simulator_sde.py`: expression lambdification and fingerprint evaluation.
- `sde_dataset_generator.py`: role-token encoding and expression normalization.

## Do Not Touch First

- Do not make `active_distance` or `state_dependent_drift` the default.
- Do not run more broad active/weak weight grids.
- Do not redesign `multi_active_weak_v1`.
- Do not change `<DRIFT> ... <DIFFUSION> ...`.
- Do not retrain.
- Do not overwrite data pickles or checkpoints.
- Do not run 64-sample expansion until a 32-sample setting improves selected
  recovery again.

## Verification Order

1. `py_compile` edited Python files.
2. `git diff --check`.
3. For helper-only diagnostics, run the helper and redirect output to
   `/private/tmp/<descriptive>.log`.
4. 8- or 16-sample eval-only smoke if probe debug/output changes.
5. 32-sample eval-only probe from:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

## Success Standard

The next useful decoding change should exceed the current role-wise baseline of
`5/32` selected exact/relaxed on the same 32-sample setup, or clearly explain
why feature normalization/fingerprint ambiguity prevents scorer-only progress
and candidate generation must be improved before scoring can move further.
