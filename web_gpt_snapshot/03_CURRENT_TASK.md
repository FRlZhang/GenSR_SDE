# Current Task

## Next Engineering Task

Validate the successful near-tie rerank tie-breaks on larger held-out probes.

At the end of that workflow, update `web_gpt_snapshot/02_RECENT_CHANGES.md` and
`web_gpt_snapshot/03_CURRENT_TASK.md`. Also update `01_PROJECT_STATUS.md` if the
goal, best checkpoint, key metrics, blocker, or next-step priority changes.

The reranking path, debug/oracle diagnostics, grammar-constrained stochastic
sampling, and drift/diffusion pairing are implemented and verified. A 16-sample
eval-only checkpoint probe with pairing produced nonzero oracle exact/relaxed
hits (`1/16`), but selected reranked exact/relaxed metrics remained zero.
Componentwise scoring improved the inspected oracle candidate from rank 11 to
rank 8. Constant-grid componentwise improved it further to rank 5 and reduced
its distance from 5.919984 to 5.125097 with `best_constant=0.5`, but selected
reranked exact/relaxed metrics remained zero.
`constant_grid_componentwise_no_multi_u0` moved the oracle to rank 2 with
distance 1.507198, just behind a top non-oracle at 1.505882. The top non-oracle
keeps the oracle diffusion but uses a constant-only drift template.
Adding `--rerank-tie-epsilon 0.005` plus either `--rerank-tie-break
active_distance` or `--rerank-tie-break state_dependent_drift` selected that
oracle and produced the first selected reranked exact/relaxed hit (`1/16`).

## Primary File To Modify

```text
sde_validation_probe.py
```

Likely next additions:

- re-run `constant_grid_componentwise_no_multi_u0` with `active_distance` and
  `state_dependent_drift` tie-breaks on 32/64 samples;
- compare false positives and whether selected exact/relaxed remains above 0;
- keep constant-grid active+weak scoring plus epsilon tie-breaks as the strongest
  current diagnostic;
- keep pairing enabled while controlling candidate counts for CPU cost;
- keep reporting reranked metrics beside greedy and constrained beam.

## Supporting Files If Needed

- `sde_fingerprint.py`: fingerprint computation and distance design.
- `simulator_sde.py`: `SDESystem`, `solve_fingerprint`, expression lambdification.
- `sde_dataset_generator.py`: role-token encoding and expression normalization.
- `train.py`: only if touching the full training path.

## Do Not Touch First

- Do not redesign `multi_active_weak_v1` fingerprint before testing reranking.
- Do not switch to two fully independent decoders by default.
- Do not change the target format away from `<DRIFT> ... <DIFFUSION> ...`.
- Do not rely on training loss alone as evidence.
- Do not overwrite data pickles or checkpoints.

## Verification Order

1. `py_compile` edited Python files.
2. `git diff --check`.
3. Small `sde_validation_probe.py` smoke test.
4. Eval-only held-out decoding from:

```text
/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth
```

## Metrics To Compare

Report each for greedy, constrained beam, and reranked candidates:

- `sequence_exact`
- `sequence_relaxed_no_constants`
- `sequence_nonempty`
- `sequence_avg_len`
- `sequence_structural_token_frac`
- `rerank_oracle_sequence_exact`
- `rerank_oracle_sequence_relaxed_no_constants`

## Success Standard

Reranking should improve `sequence_exact` or `sequence_relaxed_no_constants` over greedy and constrained beam on the same held-out setup.
