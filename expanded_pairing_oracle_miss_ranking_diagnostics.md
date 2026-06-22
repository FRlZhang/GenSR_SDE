# Expanded Pairing Oracle-Miss Ranking Diagnostics

Date: 2026-06-22

Scope: log-parse-only diagnosis from `/private/tmp/gensr_sde_32_expanded_pairing_rolewise.log`. No eval, retraining, candidate regeneration, or scorer change was run.

## Executive Summary

- Selected misses analyzed: `12`.
- Oracle-only-pair misses: `7`.
- Beam/sampling oracle score misses: `5`.
- Score gaps: `2` near (`<=0.10`), `4` moderate, `6` large (`>0.25`).
- Pair-only score gaps: `1` near and `2` large.
- Average score gap: `0.263465` overall, `0.240347` for oracle-only-pair misses.

The newly introduced paired oracles usually lose because the active+weak role-wise score assigns a lower distance to a non-oracle. This is not mostly a tie-break problem: only two of the twelve misses are within `0.10`, and only one of the seven oracle-only-pair misses is within `0.10`.

For oracle-only-pair misses, the selected candidate's advantage is dominated by `active_kramers_moyal` in all seven cases. That points to scorer/residual behavior around paired drift/diffusion combinations, not just candidate availability.

## Aggregate Diagnosis

| Diagnostic | Count |
| --- | ---: |
| Selected misses | 12 |
| Oracle only appears from pair | 7 |
| Oracle from beam/sampling but score loses | 5 |
| Same diffusion, wrong drift | 4 |
| Same drift, wrong diffusion | 4 |
| Both drift and diffusion wrong | 4 |
| Oracle has lower model score | 5 |
| Oracle-only-pair and lower model score | 1 |
| Selected source is pair | 8 |
| Selected source is beam | 4 |

Model score conflict is not systematic for the oracle-only-pair misses: only one of seven pair-only oracle misses has a lower oracle model score. It is more visible in non-pair score misses, where four of five have a lower oracle model score. Since this eval selected by rerank distance, not by model score except in tie logic, model-score preference alone does not explain the expanded-pairing regression.

## Per-Sample Miss Table

| Sample | Selected source | Oracle source | Pair-only oracle | Miss side | Score gap | Dominant selected advantage | Selected model | Oracle model | Same drift | Same diffusion |
| ---: | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| 5 | beam | pair | yes | drift | 0.209579 | active_kramers_moyal | -0.615544 | -0.246277 | no | yes |
| 8 | beam | pair | yes | drift | 0.088940 | active_kramers_moyal | -0.615708 | -0.246400 | no | yes |
| 11 | pair | pair | yes | drift | 0.142660 | active_kramers_moyal | -0.246568 | -0.213037 | no | yes |
| 12 | pair | pair | yes | drift | 0.140340 | active_kramers_moyal | -0.191778 | -0.153975 | no | yes |
| 13 | pair | sample_t=1.2 | no | both | 0.490017 | gaussian_weak_kernel | -0.160274 | -0.714439 | no | no |
| 17 | pair | beam | no | diffusion | 0.061494 | active_kramers_moyal | -0.196011 | -0.628291 | yes | no |
| 19 | beam | beam | no | both | 0.324574 | active_kramers_moyal | -0.627859 | -0.536391 | no | no |
| 21 | pair | beam | no | diffusion | 0.287712 | gaussian_weak_kernel | -0.179949 | -0.628037 | yes | no |
| 24 | pair | pair | yes | both | 0.553748 | active_kramers_moyal | -0.196258 | -0.168960 | no | no |
| 26 | pair | beam | no | both | 0.315351 | gaussian_weak_kernel | -0.196324 | -0.536036 | no | no |
| 30 | pair | pair | yes | diffusion | 0.186184 | active_kramers_moyal | -0.195961 | -0.196413 | yes | no |
| 31 | beam | pair | yes | diffusion | 0.360978 | active_kramers_moyal | -0.651284 | -0.196033 | yes | no |

## Per-Sample Details

### Sample 5

- Truth: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`
- Selected: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`
- Best oracle: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`
- Sources: selected `beam`, oracle `pair`, oracle-only-pair `yes`.
- Scores: selected `1.058023`, oracle `1.267602`, gap `0.209579`.
- Model scores: selected `-0.615544`, oracle `-0.246277`, oracle lower `no`.
- Selected drift: `mul mul CONSTANT CONSTANT CONSTANT`
- Oracle drift: `mul CONSTANT CONSTANT`
- Selected diffusion: `mul CONSTANT x_0`
- Oracle diffusion: `mul CONSTANT x_0`
- Role-wise constants: selected drift `1`, selected diffusion `0.5`, oracle drift `1`, oracle diffusion `0.5`.
- Active distances: selected `0.729762`, oracle `0.932397`.
- Weak distances: selected `0.328261`, oracle `0.335205`.
- Match flags: same drift `no`, same diffusion `yes`, miss side `drift`.

### Sample 8

- Truth: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`
- Selected: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`
- Best oracle: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`
- Sources: selected `beam`, oracle `pair`, oracle-only-pair `yes`.
- Scores: selected `0.406975`, oracle `0.495915`, gap `0.088940`.
- Model scores: selected `-0.615708`, oracle `-0.246400`, oracle lower `no`.
- Selected drift: `mul mul CONSTANT CONSTANT CONSTANT`
- Oracle drift: `mul CONSTANT CONSTANT`
- Selected diffusion: `mul CONSTANT x_0`
- Oracle diffusion: `mul CONSTANT x_0`
- Role-wise constants: selected drift `2`, selected diffusion `1`, oracle drift `2`, oracle diffusion `0.5`.
- Active distances: selected `0.144263`, oracle `0.236728`.
- Weak distances: selected `0.262711`, oracle `0.259187`.
- Match flags: same drift `no`, same diffusion `yes`, miss side `drift`.

### Sample 11

- Truth: `<DRIFT> mul mul CONSTANT CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`
- Selected: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> mul CONSTANT x_0`
- Best oracle: `<DRIFT> mul mul CONSTANT CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`
- Sources: selected `pair`, oracle `pair`, oracle-only-pair `yes`.
- Scores: selected `0.699177`, oracle `0.841837`, gap `0.142660`.
- Model scores: selected `-0.246568`, oracle `-0.213037`, oracle lower `no`.
- Selected drift: `mul CONSTANT CONSTANT`
- Oracle drift: `mul mul CONSTANT CONSTANT x_0`
- Selected diffusion: `mul CONSTANT x_0`
- Oracle diffusion: `mul CONSTANT x_0`
- Role-wise constants: selected drift `0.25`, selected diffusion `1`, oracle drift `0.25`, oracle diffusion `1`.
- Active distances: selected `0.265916`, oracle `0.396132`.
- Weak distances: selected `0.433261`, oracle `0.445705`.
- Match flags: same drift `no`, same diffusion `yes`, miss side `drift`.

### Sample 12

- Truth: `<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`
- Selected: `<DRIFT> mul CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`
- Best oracle: `<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`
- Sources: selected `pair`, oracle `pair`, oracle-only-pair `yes`.
- Scores: selected `0.988306`, oracle `1.128645`, gap `0.140340`.
- Model scores: selected `-0.191778`, oracle `-0.153975`, oracle lower `no`.
- Selected drift: `mul CONSTANT CONSTANT`
- Oracle drift: `mul mul CONSTANT CONSTANT sin x_0`
- Selected diffusion: `add CONSTANT mul CONSTANT abs x_0`
- Oracle diffusion: `add CONSTANT mul CONSTANT abs x_0`
- Role-wise constants: selected drift `0.25`, selected diffusion `1`, oracle drift `0.25`, oracle diffusion `1`.
- Active distances: selected `0.465176`, oracle `0.616531`.
- Weak distances: selected `0.523130`, oracle `0.512114`.
- Match flags: same drift `no`, same diffusion `yes`, miss side `drift`.

### Sample 13

- Truth: `<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> mul CONSTANT CONSTANT`
- Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`
- Best oracle: `<DRIFT> mul mul CONSTANT CONSTANT sin x_0 <DIFFUSION> mul CONSTANT CONSTANT`
- Sources: selected `pair`, oracle `sample_t=1.2`, oracle-only-pair `no`.
- Scores: selected `1.184112`, oracle `1.674129`, gap `0.490017`.
- Model scores: selected `-0.160274`, oracle `-0.714439`, oracle lower `yes`.
- Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Oracle drift: `mul mul CONSTANT CONSTANT sin x_0`
- Selected diffusion: `add CONSTANT mul CONSTANT abs x_0`
- Oracle diffusion: `mul CONSTANT CONSTANT`
- Role-wise constants: selected drift `0.25`, selected diffusion `0.25`, oracle drift `0.25`, oracle diffusion `1`.
- Active distances: selected `0.537799`, oracle `0.730690`.
- Weak distances: selected `0.646314`, oracle `0.943439`.
- Match flags: same drift `no`, same diffusion `no`, miss side `both`.

### Sample 17

- Truth: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`
- Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`
- Best oracle: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`
- Sources: selected `pair`, oracle `beam`, oracle-only-pair `no`.
- Scores: selected `0.756648`, oracle `0.818141`, gap `0.061494`.
- Model scores: selected `-0.196011`, oracle `-0.628291`, oracle lower `yes`.
- Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Oracle drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Selected diffusion: `mul CONSTANT x_0`
- Oracle diffusion: `mul CONSTANT sqrt abs x_0`
- Role-wise constants: selected drift `0.25`, selected diffusion `0.5`, oracle drift `1`, oracle diffusion `0.5`.
- Active distances: selected `0.374953`, oracle `0.536178`.
- Weak distances: selected `0.381695`, oracle `0.281964`.
- Match flags: same drift `yes`, same diffusion `no`, miss side `diffusion`.

### Sample 19

- Truth: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`
- Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`
- Best oracle: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`
- Sources: selected `beam`, oracle `beam`, oracle-only-pair `no`.
- Scores: selected `1.578646`, oracle `1.903220`, gap `0.324574`.
- Model scores: selected `-0.627859`, oracle `-0.536391`, oracle lower `no`.
- Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Oracle drift: `mul mul CONSTANT CONSTANT CONSTANT`
- Selected diffusion: `mul CONSTANT sqrt abs x_0`
- Oracle diffusion: `add CONSTANT mul CONSTANT abs x_0`
- Role-wise constants: selected drift `0.25`, selected diffusion `0.5`, oracle drift `0.25`, oracle diffusion `0.25`.
- Active distances: selected `0.721302`, oracle `0.941829`.
- Weak distances: selected `0.857344`, oracle `0.961391`.
- Match flags: same drift `no`, same diffusion `no`, miss side `both`.

### Sample 21

- Truth: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`
- Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT abs x_0`
- Best oracle: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT sqrt abs x_0`
- Sources: selected `pair`, oracle `beam`, oracle-only-pair `no`.
- Scores: selected `0.514409`, oracle `0.802121`, gap `0.287712`.
- Model scores: selected `-0.179949`, oracle `-0.628037`, oracle lower `yes`.
- Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Oracle drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Selected diffusion: `mul CONSTANT abs x_0`
- Oracle diffusion: `mul CONSTANT sqrt abs x_0`
- Role-wise constants: selected drift `1`, selected diffusion `0.25`, oracle drift `1`, oracle diffusion `0.25`.
- Active distances: selected `0.187390`, oracle `0.231566`.
- Weak distances: selected `0.327019`, oracle `0.570555`.
- Match flags: same drift `yes`, same diffusion `no`, miss side `diffusion`.

### Sample 24

- Truth: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT sqrt abs x_0`
- Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT`
- Best oracle: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> mul CONSTANT sqrt abs x_0`
- Sources: selected `pair`, oracle `pair`, oracle-only-pair `yes`.
- Scores: selected `1.465351`, oracle `2.019098`, gap `0.553748`.
- Model scores: selected `-0.196258`, oracle `-0.168960`, oracle lower `no`.
- Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Oracle drift: `mul mul CONSTANT CONSTANT CONSTANT`
- Selected diffusion: `mul CONSTANT CONSTANT`
- Oracle diffusion: `mul CONSTANT sqrt abs x_0`
- Role-wise constants: selected drift `0.25`, selected diffusion `0.5`, oracle drift `0.25`, oracle diffusion `0.5`.
- Active distances: selected `0.596446`, oracle `1.071399`.
- Weak distances: selected `0.868905`, oracle `0.947699`.
- Match flags: same drift `no`, same diffusion `no`, miss side `both`.

### Sample 26

- Truth: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`
- Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT`
- Best oracle: `<DRIFT> mul mul CONSTANT CONSTANT CONSTANT <DIFFUSION> add CONSTANT mul CONSTANT abs x_0`
- Sources: selected `pair`, oracle `beam`, oracle-only-pair `no`.
- Scores: selected `1.130573`, oracle `1.445924`, gap `0.315351`.
- Model scores: selected `-0.196324`, oracle `-0.536036`, oracle lower `yes`.
- Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Oracle drift: `mul mul CONSTANT CONSTANT CONSTANT`
- Selected diffusion: `mul CONSTANT CONSTANT`
- Oracle diffusion: `add CONSTANT mul CONSTANT abs x_0`
- Role-wise constants: selected drift `0.25`, selected diffusion `1`, oracle drift `0.25`, oracle diffusion `0.5`.
- Active distances: selected `0.609797`, oracle `0.674083`.
- Weak distances: selected `0.520776`, oracle `0.771841`.
- Match flags: same drift `no`, same diffusion `no`, miss side `both`.

### Sample 30

- Truth: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT`
- Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`
- Best oracle: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT CONSTANT`
- Sources: selected `pair`, oracle `pair`, oracle-only-pair `yes`.
- Scores: selected `0.403688`, oracle `0.589872`, gap `0.186184`.
- Model scores: selected `-0.195961`, oracle `-0.196413`, oracle lower `yes`.
- Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Oracle drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Selected diffusion: `mul CONSTANT x_0`
- Oracle diffusion: `mul CONSTANT CONSTANT`
- Role-wise constants: selected drift `1`, selected diffusion `0.25`, oracle drift `1`, oracle diffusion `0.5`.
- Active distances: selected `0.159342`, oracle `0.376491`.
- Weak distances: selected `0.244346`, oracle `0.213380`.
- Match flags: same drift `yes`, same diffusion `no`, miss side `diffusion`.

### Sample 31

- Truth: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`
- Selected: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT abs x_0`
- Best oracle: `<DRIFT> mul CONSTANT mul x_0 sub CONSTANT x_0 <DIFFUSION> mul CONSTANT x_0`
- Sources: selected `beam`, oracle `pair`, oracle-only-pair `yes`.
- Scores: selected `0.596731`, oracle `0.957709`, gap `0.360978`.
- Model scores: selected `-0.651284`, oracle `-0.196033`, oracle lower `no`.
- Selected drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Oracle drift: `mul CONSTANT mul x_0 sub CONSTANT x_0`
- Selected diffusion: `mul CONSTANT abs x_0`
- Oracle diffusion: `mul CONSTANT x_0`
- Role-wise constants: selected drift `1`, selected diffusion `0.5`, oracle drift `1`, oracle diffusion `0.5`.
- Active distances: selected `0.342968`, oracle `0.667490`.
- Weak distances: selected `0.253762`, oracle `0.290219`.
- Match flags: same drift `yes`, same diffusion `no`, miss side `diffusion`.

## Recommendation

Do not make expanded pairing the default and do not run 64-sample expansion from this setting yet.

A simple pair-aware tie-break is not justified from this log alone. Most paired oracles do not lose by epsilon-sized gaps; they lose because active+weak distance prefers non-oracle candidates. A large pair bonus would be needed for several cases and would likely overfit or promote wrong pair candidates.

Candidate pruning or pair-aware ranking diagnostics are more justified than a blanket pair bonus. Wrong pair candidates are selected in eight of the twelve misses, and expanded pairing increases both useful and harmful paired candidates. The next low-cost step should inspect the top paired non-oracles and compare their active/weak residuals against the paired oracles, especially for the seven oracle-only-pair misses.
