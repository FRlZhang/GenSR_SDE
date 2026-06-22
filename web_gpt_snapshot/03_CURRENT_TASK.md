# Current Task

## Next Engineering Task

Analyze expanded-pairing oracle-miss ranking from existing logs. Expanded
pairing raised oracle coverage, but selected recovery regressed, so the next
question is why the scorer/ranker misses newly introduced paired oracles.

Current strongest no-retraining decoding setting:

```text
rerank_score=constant_grid_rolewise_no_multi_u0
active:weak=1:1
selected exact/relaxed=5/32
oracle exact/relaxed=9/32
pair oracle exact/relaxed=2/32
```

Expanded-pairing formal eval:

```text
pair_drift_topk=5
pair_diffusion_topk=6
pair_drift_diffusion_candidates=32
selected exact/relaxed=3/32
oracle exact/relaxed=15/32
pair oracle exact/relaxed=8/32
oracle-present samples=15
selected oracle hits=3
selected oracle misses=12
oracle-only-pair misses=7
beam/sampling score misses=5
```

Candidate-coverage-only result:

```text
expanded pairing: pair_drift_topk=5, pair_diffusion_topk=6, pair_drift_diffusion_candidates=32
full_oracle_present=15/32
oracle_absent=17/32
rescued pairing-missing samples=5,8,11,23,30,31
```

Role-wise constants rescued one shared-constant miss, sample 17, by allowing
drift constant `1.0` and diffusion constant `0.5`. The active-heavy `2:1` and
weak-downweighted `1:0.5` variants both regressed to `3/32`, so do not continue
broad active/weak grid search.

Targeted residual-vector diagnostics now exist:

```text
scripts/analyze_rolewise_residuals.py
rolewise_residual_vector_diagnostics.md
rolewise_residual_vector_diagnostics.json
/private/tmp/gensr_sde_rolewise_residual_vectors.log
```

Residual-vector classifications:

```text
sample 13: both-side ambiguity; feature-scaling artifact; model-score conflict
sample 19: both-side ambiguity; feature-scaling artifact
sample 24: drift-side ambiguity; feature-scaling artifact
sample 26: diffusion-side ambiguity; feature-scaling artifact
```

Offline score ablation also exists:

```text
scripts/analyze_residual_score_ablation.py
rolewise_residual_score_ablation.md
/private/tmp/gensr_sde_residual_score_ablation.log
```

Key result: mild robust/normalization variants rescued `0/4`; only aggressive
`clipped_l2_p90` flipped sample 19. Samples 24 and 26 did not flip under any
tested offline score. Do not add a robust-scoring rerank mode from this evidence
alone.

Candidate coverage diagnostics now exist:

```text
scripts/analyze_candidate_coverage.py
candidate_coverage_diagnostics.md
/private/tmp/gensr_sde_candidate_coverage.log
```

Key result:

```text
full_oracle_present=9/32
oracle_absent=23/32
diffusion_only_present=17
drift_and_diffusion_separately_present_but_not_paired=6
drift_only_present=0
neither_side_present=0
```

Interpretation: prioritize drift span diversity first, then pairing
coverage/ranking. Diffusion coverage is already broad in this candidate pool.

Drift/pairing taxonomy exists:

```text
scripts/analyze_drift_pairing_coverage.py
drift_pairing_coverage_diagnostics.md
/private/tmp/gensr_sde_drift_pairing_coverage_full.log
```

Available taxonomy and rank diagnosis:

```text
linear drift missing=6
sin drift missing=6
nested-mul linear drift missing=3
polynomial-like drift missing=2
pairing-missing samples=5,8,11,23,30,31
pairing actions after expansion=all six rescued as pair-source full oracles
```

Pairing details: samples `5`, `8`, `11`, `30`, and `31` were rescued by the
larger pair cap. Sample `23` was rescued by `pair_drift_topk=5` plus the larger
cap; its combined pair rank was 24.

## Primary File To Modify

No primary source file needs to change unless the next task explicitly adds a
new log parser. Prefer a narrow helper under `scripts/` and existing log data
before modifying `sde_validation_probe.py`.

Likely next additions:

- parse expanded-pairing miss cases from
  `/private/tmp/gensr_sde_32_expanded_pairing_rolewise.log`;
- compare selected non-oracle vs best paired oracle score, source, drift,
  diffusion, constants, and model score for the 7 oracle-only-pair misses;
- decide whether a pair-aware ranking heuristic, pair source penalty/bonus,
  model-score tie-break, or candidate pruning is worth testing;
- defer drift span diversity work until the expanded-pairing ranking failure is
  understood;
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
- Do not make expanded pairing the default: formal selected exact/relaxed
  regressed to `3/32` even though oracle rose to `15/32`.

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

The next useful step should explain the 12 expanded-pairing selected misses,
especially the 7 oracle-only-pair misses, using existing logs before launching
new experiments.
