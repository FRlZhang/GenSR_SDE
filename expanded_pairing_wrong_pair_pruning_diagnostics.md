# Expanded Pairing Wrong-Pair Pruning Diagnostics

Date: 2026-06-22

Scope: existing-log/JSON analysis only. No model run, candidate regeneration, formal eval, retraining, scorer change, or fingerprint change was performed.

## Executive Summary

- Selected misses analyzed: `12`.
- Wrong selected pair cases: `8`.
- Pair-oracle samples that must be preserved: `8`.
- Pair-oracle source ranks span `2` to `15`.
- Approximate wrong-pair source ranks span `1` to `14`.
- Active-distance traps among wrong selected pairs: `5/8`.

No safe pruning criterion is visible from the existing logs. The pair oracles are often late pair-source candidates, so rank/cap pruning would remove real oracle pairs. Structural pruning is also unsafe: harmful mean-reverting or constant-drift pairs overlap with true pair-oracle families. The dominant failure is scorer behavior, especially active-kramers-moyal traps, not obviously invalid pair structure.

Outcome: **B. No safe pruning criterion exists from these logs; recommend scorer/residual analysis of active_kramers_moyal traps before testing a heuristic.**

## Wrong Selected Pair Cases

| Sample | Oracle source | Pair-only oracle | Miss side | Selected pair rank approx | Oracle pair rank | Score gap | Active gap | Weak gap | Selected drift | Selected diffusion |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 11 | pair | yes | drift | 4 | 10 | 0.142660 | 0.130216 | 0.012444 | `mul CONSTANT CONSTANT` | `mul CONSTANT x_0` |
| 12 | pair | yes | drift | 1 | 2 | 0.140340 | 0.151355 | -0.011016 | `mul CONSTANT CONSTANT` | `add CONSTANT mul CONSTANT abs x_0` |
| 13 | sample_t=1.2 | no | both | 8 | None | 0.490017 | 0.192891 | 0.297125 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `add CONSTANT mul CONSTANT abs x_0` |
| 17 | beam | no | diffusion | 10 | None | 0.061494 | 0.161225 | -0.099731 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT x_0` |
| 21 | beam | no | diffusion | 14 | None | 0.287712 | 0.044176 | 0.243536 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT abs x_0` |
| 24 | pair | yes | both | 14 | 2 | 0.553748 | 0.474953 | 0.078794 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT CONSTANT` |
| 26 | beam | no | both | 14 | None | 0.315351 | 0.064286 | 0.251065 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT CONSTANT` |
| 30 | pair | yes | diffusion | 10 | 10 | 0.186184 | 0.217149 | -0.030966 | `mul CONSTANT mul x_0 sub CONSTANT x_0` | `mul CONSTANT x_0` |

Positive active/weak gap means the selected wrong pair has a lower distance than the oracle for that segment. Most wrong selected pairs are not parse-invalid-looking; they are plausible recombinations that the current active+weak scorer prefers.

## Pair-Oracle Preservation Checks

| Pair-oracle sample | Pair source rank | Global candidate rank | Model score |
| ---: | ---: | ---: | ---: |
| 5 | 12 | 22 | -0.246277 |
| 8 | 13 | 22 | -0.246400 |
| 11 | 10 | 20 | -0.213037 |
| 12 | 2 | 12 | -0.153975 |
| 23 | 15 | 24 | -0.196104 |
| 24 | 2 | 11 | -0.168960 |
| 30 | 10 | 20 | -0.196413 |
| 31 | 9 | 19 | -0.196033 |

Simple pair source-rank pruning is unsafe because preserving all pair oracles requires keeping ranks up to 15, which also keeps the observed wrong selected pairs.

| Candidate rule | Removes pair-oracle samples | Preserves all pair oracles |
| --- | --- | --- |
| `keep_pair_source_rank<=2` | 5,8,11,23,30,31 | no |
| `keep_pair_source_rank<=4` | 5,8,11,23,30,31 | no |
| `keep_pair_source_rank<=6` | 5,8,11,23,30,31 | no |
| `keep_pair_source_rank<=8` | 5,8,11,23,30,31 | no |
| `keep_pair_source_rank<=10` | 5,8,23 | no |
| `keep_pair_source_rank<=12` | 8,23 | no |
| `keep_pair_source_rank<=15` | none | yes |

## Pruning Interpretation

- Harmful high-ranking pair candidates are parse-valid and structurally plausible; they do not look like malformed equations.
- Wrong selected pairs split across wrong-drift, wrong-diffusion, and both-side errors, so no one-sided structural filter is clean.
- Mean-reverting selected pair drift appears in `6` wrong selected pair cases, but mean-reverting drift is also the true drift for paired-oracle samples 23, 30, and 31.
- Constant selected pair drift appears in `2` wrong selected pair cases, but constant drift is the true drift for paired-oracle samples 5 and 8.
- Pair source rank is not a safe pruning signal: oracle pairs can appear at source ranks 9, 10, 12, 13, and 15.
- Model score is not a clean pruning signal either; wrong pair and oracle pair model scores overlap tightly around the pair-candidate range.
- Score-component pruning is risky because wrong pairs are often active-distance traps: they win the active segment even when they are symbolically wrong.

## Recommendation

Do not add pair candidate pruning or a pair bonus from the current evidence. The safer next step is a targeted residual analysis of active_kramers_moyal traps for wrong pair candidates versus pair oracles, ideally reusing existing residual-vector tooling or adding a log-only/top-candidate residual parser if enough fields are present.

Only if that analysis identifies a concrete, oracle-preserving rule should a future small/offline pruning diagnostic be proposed. Do not run a formal eval or 64-sample expansion for pruning yet.
