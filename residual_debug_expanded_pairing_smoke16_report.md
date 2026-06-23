# Expanded Pairing Residual Debug Smoke 16 Report

Date: 2026-06-23

## Status

Blocked before execution by the workflow time limit. I did not run the
expanded-pairing 16-sample smoke or the safety helper.

Reason: the previous baseline 16-sample residual-debug smoke took about
9.5 minutes with 201 logged candidates. Expanded pairing roughly increases the
candidate pool by the same ratio seen in the 32-sample formal run
(`715 / 405`, about 1.77x), so the expanded 16-sample smoke is expected to
exceed the requested 10-minute Codex limit.

## Prepared Local Script

The smallest local command is:

```bash
bash scripts/run_expanded_pairing_residual_debug_smoke16.sh
```

The script runs exactly:

1. one 16-sample expanded-pairing residual-debug smoke;
2. one offline safety parse with `scripts/analyze_residual_debug_safety.py`.

It writes logs and JSON to:

```text
/private/tmp/gensr_sde_residual_debug_expanded_pairing_smoke16.log
/private/tmp/gensr_sde_residual_debug_expanded_pairing_smoke16.json
/private/tmp/gensr_sde_residual_debug_safety_expanded_pairing_smoke16.log
residual_debug_safety_expanded_pairing_smoke16.md
residual_debug_safety_expanded_pairing_smoke16.json
```

## Requested Metrics

No new smoke metrics are available because the smoke was not run.

| Item | Value |
| --- | --- |
| Smoke completed | no |
| Samples logged | n/a |
| Candidates logged | n/a |
| Parse failures | n/a |
| Fingerprint failures | n/a |
| Samples with oracle candidates | n/a |
| Selected hits | n/a |
| Oracle-present selected misses | n/a |
| Pair-oracle samples | n/a |
| Active residual vectors | n/a |
| Weak residual vectors | n/a |
| Selected hits preserved | n/a |
| Selected hits harmed | n/a |
| Oracle-present selected misses rescued | n/a |
| Pair-oracle cases preserved/harmed | n/a |

## Baseline Comparison

Previous baseline 16-sample residual-debug smoke:

```text
samples logged=16
candidates logged=201
samples with oracle=2
selected hits=1
pair oracle samples=1
selected-hit harms=0
oracle-present selected-miss rescues=0
```

## Recommendation

Do not add a scorer mode and do not run a formal 32-sample eval. Run the local
script manually only if the expanded-pairing safety check is still needed. If it
shows no selected-hit harm but no rescue, stop scorer-side normalization for now
and return to drift span diversity or deeper fingerprint ambiguity diagnostics.
