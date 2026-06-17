# Current Task

## Next Engineering Task

Implement diverse candidate generation plus SDE fingerprint-distance reranking.

At the end of that workflow, update `web_gpt_snapshot/02_RECENT_CHANGES.md` and
`web_gpt_snapshot/03_CURRENT_TASK.md`. Also update `01_PROJECT_STATUS.md` if the
goal, best checkpoint, key metrics, blocker, or next-step priority changes.

The handoff system is now ready for commit: `web_gpt_snapshot/` is the stable
lightweight snapshot to keep, while `web_gpt_handoff/` is ignored as a local
temporary package.

## Primary File To Modify

```text
sde_validation_probe.py
```

Likely additions:

- generate multiple valid candidate sequences per sample;
- parse candidates into drift/diffusion expressions;
- reject invalid expressions without crashing;
- compute candidate SDE fingerprints;
- rank by distance to the target held-out fingerprint;
- report reranked metrics beside greedy and constrained beam.

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

## Success Standard

Reranking should improve `sequence_exact` or `sequence_relaxed_no_constants` over greedy and constrained beam on the same held-out setup.
