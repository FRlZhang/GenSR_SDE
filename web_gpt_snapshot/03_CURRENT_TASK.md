# Current Task

## Next Engineering Task

Close the rerank selected-versus-oracle gap on the 32-sample eval-only probe.

The near-tie tie-break validation is complete. With the 2000-step checkpoint,
`constant_grid_componentwise_no_multi_u0`, `--rerank-tie-epsilon 0.005`,
beam+sampling candidates, and drift/diffusion pairing, all three 32-sample
settings matched:

```text
baseline no tie selected exact/relaxed=4/32
active_distance selected exact/relaxed=4/32
state_dependent_drift selected exact/relaxed=4/32
oracle exact/relaxed=9/32
pair oracle exact/relaxed=2/32
```

Because neither tie-break exceeded the no-tie baseline, 64-sample validation was
not run. The current bottleneck is not a missing near-tie rule; it is that the
candidate pool contains more correct templates than the score selects.

## Primary File To Modify

```text
sde_validation_probe.py
```

Likely next additions:

- inspect the 32-sample cases where oracle exists but selected rerank misses;
- compare selected versus oracle constants and active/weak segment distances;
- improve candidate generation or drift/diffusion pairing coverage before
  adding another tie-break heuristic;
- consider more targeted constant handling, since fixed/global constants still
  distort ranking;
- keep reporting reranked metrics beside greedy, constrained beam, oracle, and
  pair-oracle metrics.

## Supporting Files If Needed

- `sde_fingerprint.py`: fingerprint computation and distance design.
- `simulator_sde.py`: `SDESystem`, `solve_fingerprint`, expression lambdification.
- `sde_dataset_generator.py`: role-token encoding and expression normalization.
- `train.py`: only if touching the full training path.

## Do Not Touch First

- Do not make `active_distance` or `state_dependent_drift` the default based on
  the earlier 16-sample hit.
- Do not redesign `multi_active_weak_v1` fingerprint before exhausting decoding
  and reranking diagnostics.
- Do not switch to two fully independent decoders by default.
- Do not change the target format away from `<DRIFT> ... <DIFFUSION> ...`.
- Do not rely on training loss alone as evidence.
- Do not overwrite data pickles or checkpoints.

## Verification Order

1. `py_compile` edited Python files.
2. `git diff --check`.
3. Small `sde_validation_probe.py` smoke test if code changes.
4. Eval-only held-out decoding from:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

## Metrics To Compare

Report each for greedy, constrained beam, reranked candidates, oracle, and pair
oracle:

- `sequence_exact`
- `sequence_relaxed_no_constants`
- `rerank_candidates_attempted`
- `rerank_candidates_valid`
- `rerank_fingerprint_failures`
- `rerank_unique_candidate_avg`
- `rerank_unique_drift_avg`
- `rerank_unique_diffusion_avg`
- `rerank_unique_paired_candidate_avg`

## Success Standard

The next decoding change should improve selected reranked exact or relaxed
recovery beyond the current 32-sample no-tie baseline of `4/32`, while staying
below and explaining the oracle ceiling of `9/32`.
