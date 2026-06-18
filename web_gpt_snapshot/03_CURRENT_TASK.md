# Current Task

## Next Engineering Task

Use the oracle-gap diagnostics to improve selected reranking from `4/32` toward
the current oracle ceiling of `9/32`.

The 32-sample no-tie eval-only rerun from
`/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth` reproduced the
previous metrics and added miss classification:

```text
selected exact/relaxed=4/32
oracle exact/relaxed=9/32
pair oracle exact/relaxed=2/32
oracle misses=5
parse_failures=0
fingerprint_failures=0
```

Oracle sources:

```text
best oracle sources: beam=6 sampling=1 pair=2
selected hit sources: beam=3 pair=1
```

Miss types:

```text
same diffusion but wrong drift=2
same drift but wrong diffusion=1
both drift and diffusion wrong=2
constant-only mismatch=0
oracle lower model-score candidate=2
oracle only appears from pair=1
oracle from beam/sampling but score misses=4
```

The immediate problem is not another tie-break. In the 5 miss cases, the oracle
template is present, but current active/weak constant-grid scoring prefers a
non-oracle. The larger ceiling remains candidate generation because 23/32
samples still have no oracle candidate.

## Primary File To Modify

```text
sde_validation_probe.py
```

Likely next additions:

- compare selected versus oracle score components in the 5 miss cases;
- test score calibration or constant-handling diagnostics without changing the
  target format, training path, or fingerprint schema;
- inspect whether active versus weak weighting is over-rewarding wrong drift or
  wrong diffusion templates;
- preserve oracle-gap summary output so each change reports selected, oracle,
  source, and miss-type counts.

## Supporting Files If Needed

- `sde_fingerprint.py`: only for reading score component semantics; do not
  redesign the schema first.
- `simulator_sde.py`: expression lambdification and fingerprint evaluation.
- `sde_dataset_generator.py`: role-token encoding and expression normalization.

## Do Not Touch First

- Do not make `active_distance` or `state_dependent_drift` the default.
- Do not redesign `multi_active_weak_v1` before exhausting score and constant
  diagnostics.
- Do not change `<DRIFT> ... <DIFFUSION> ...`.
- Do not retrain for this diagnostic loop.
- Do not overwrite data pickles or checkpoints.
- Do not run 64-sample expansion until a 32-sample setting improves selected
  recovery over `4/32`.

## Verification Order

1. `py_compile` edited Python files.
2. `git diff --check`.
3. 8- or 16-sample eval-only smoke if debug/output changes.
4. 32-sample eval-only probe from:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

## Success Standard

Improve selected reranked exact or relaxed recovery above `4/32` on the same
32-sample setup, while reporting whether the oracle ceiling remains `9/32` or
candidate generation changed.
