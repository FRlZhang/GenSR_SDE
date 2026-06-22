#!/usr/bin/env python3
"""Parse expanded-pairing oracle miss diagnostics from an existing probe log."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-log",
        type=Path,
        default=Path("/private/tmp/gensr_sde_32_expanded_pairing_rolewise.log"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("expanded_pairing_oracle_miss_ranking_diagnostics.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("expanded_pairing_oracle_miss_ranking_diagnostics.json"),
    )
    return parser.parse_args()


def parse_key_values(text: str) -> dict[str, str]:
    pairs = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", text):
        pairs[key] = value
    return pairs


def to_number(value: str | None):
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def parse_metrics(text: str) -> dict[str, float]:
    metrics = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_]+)=([-+0-9.eE]+)$", line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def parse_oracle_misses(text: str) -> list[dict]:
    start = text.index("oracle_miss_cases:")
    end = text.index("oracle_miss_rescue_cases:", start)
    lines = text[start:end].splitlines()[1:]
    cases = []
    current = None

    def finish():
        if current is not None:
            cases.append(current)

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("sample_index="):
            finish()
            current = {"sample_index": int(line.split("=", 1)[1])}
            continue
        if current is None:
            continue
        if line.startswith("truth: "):
            current["truth_sequence"] = line.split(": ", 1)[1]
        elif line.startswith("selected_sequence: "):
            current["selected_sequence"] = line.split(": ", 1)[1]
        elif line.startswith("oracle_sequence: "):
            current["oracle_sequence"] = line.split(": ", 1)[1]
        elif line.startswith("sources "):
            values = parse_key_values(line)
            current["selected_source"] = values.get("selected")
            current["oracle_source"] = values.get("oracle")
        elif line.startswith("scores "):
            values = parse_key_values(line)
            current["selected_score"] = float(values["selected"])
            current["oracle_score"] = float(values["oracle"])
            current["score_gap"] = float(values["gap"])
        elif line.startswith("shared_constant_scores "):
            values = parse_key_values(line)
            current["shared_selected_score"] = float(values["selected"])
            current["shared_oracle_score"] = float(values["oracle"])
        elif line.startswith("best_constants "):
            values = parse_key_values(line)
            current["selected_best_constant"] = values.get("selected")
            current["oracle_best_constant"] = values.get("oracle")
        elif line.startswith("rolewise_best_constants "):
            values = parse_key_values(line)
            current["selected_best_drift_constant"] = to_number(values.get("selected_drift"))
            current["selected_best_diffusion_constant"] = to_number(
                values.get("selected_diffusion")
            )
            current["oracle_best_drift_constant"] = to_number(values.get("oracle_drift"))
            current["oracle_best_diffusion_constant"] = to_number(
                values.get("oracle_diffusion")
            )
        elif line.startswith("model_scores "):
            values = parse_key_values(line)
            current["selected_model_score"] = float(values["selected"])
            current["oracle_model_score"] = float(values["oracle"])
            current["oracle_lower_model_score"] = bool(int(values["oracle_lower"]))
        elif line.startswith("selected_drift: "):
            current["selected_drift"] = line.split(": ", 1)[1]
        elif line.startswith("oracle_drift: "):
            current["oracle_drift"] = line.split(": ", 1)[1]
        elif line.startswith("selected_diffusion: "):
            current["selected_diffusion"] = line.split(": ", 1)[1]
        elif line.startswith("oracle_diffusion: "):
            current["oracle_diffusion"] = line.split(": ", 1)[1]
        elif line.startswith("segment_distances_selected "):
            values = parse_key_values(line)
            current["selected_segment_distances"] = {
                key: float(value) for key, value in values.items()
            }
        elif line.startswith("segment_distances_oracle "):
            values = parse_key_values(line)
            current["oracle_segment_distances"] = {
                key: float(value) for key, value in values.items()
            }
        elif line.startswith("match_flags "):
            values = parse_key_values(line)
            current["shares_drift"] = bool(int(values["shares_drift"]))
            current["shares_diffusion"] = bool(int(values["shares_diffusion"]))
            current["constant_mismatch"] = bool(int(values["constant_mismatch"]))
            current["miss_side"] = values["miss_side"]
            current["oracle_only_pair"] = bool(int(values["oracle_only_pair"]))
    finish()
    return cases


def summarize(cases: list[dict]) -> dict:
    pair_only = [case for case in cases if case["oracle_only_pair"]]
    non_pair = [case for case in cases if not case["oracle_only_pair"]]
    gaps = [case["score_gap"] for case in cases]
    pair_gaps = [case["score_gap"] for case in pair_only]

    for case in cases:
        selected = case["selected_segment_distances"]
        oracle = case["oracle_segment_distances"]
        active_advantage = (
            oracle["active_kramers_moyal"] - selected["active_kramers_moyal"]
        )
        weak_advantage = (
            oracle["gaussian_weak_kernel"] - selected["gaussian_weak_kernel"]
        )
        case["active_advantage_for_selected"] = active_advantage
        case["weak_advantage_for_selected"] = weak_advantage
        case["dominant_selected_advantage"] = (
            "active_kramers_moyal"
            if abs(active_advantage) >= abs(weak_advantage)
            else "gaussian_weak_kernel"
        )

    return {
        "misses_total": len(cases),
        "oracle_only_pair_misses": len(pair_only),
        "beam_sampling_score_misses": len(non_pair),
        "near_tie_gap_le_0_10": sum(1 for gap in gaps if gap <= 0.10),
        "moderate_gap_0_10_to_0_25": sum(1 for gap in gaps if 0.10 < gap <= 0.25),
        "large_gap_gt_0_25": sum(1 for gap in gaps if gap > 0.25),
        "pair_near_tie_gap_le_0_10": sum(1 for gap in pair_gaps if gap <= 0.10),
        "pair_large_gap_gt_0_25": sum(1 for gap in pair_gaps if gap > 0.25),
        "avg_score_gap": mean(gaps),
        "median_score_gap": median(gaps),
        "avg_pair_only_score_gap": mean(pair_gaps),
        "selected_source_counts": Counter(case["selected_source"] for case in cases),
        "oracle_source_counts": Counter(case["oracle_source"] for case in cases),
        "miss_side_counts": Counter(case["miss_side"] for case in cases),
        "dominant_advantage_counts": Counter(
            case["dominant_selected_advantage"] for case in cases
        ),
        "pair_only_dominant_advantage_counts": Counter(
            case["dominant_selected_advantage"] for case in pair_only
        ),
        "oracle_lower_model_score_count": sum(
            1 for case in cases if case["oracle_lower_model_score"]
        ),
        "pair_only_oracle_lower_model_score_count": sum(
            1 for case in pair_only if case["oracle_lower_model_score"]
        ),
    }


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def write_report(path: Path, metrics: dict[str, float], cases: list[dict], summary: dict) -> None:
    lines = [
        "# Expanded Pairing Oracle-Miss Ranking Diagnostics",
        "",
        "Date: 2026-06-22",
        "",
        "Scope: log-parse-only diagnosis from `/private/tmp/gensr_sde_32_expanded_pairing_rolewise.log`. No eval, retraining, candidate regeneration, or scorer change was run.",
        "",
        "## Executive Summary",
        "",
        f"- Selected misses analyzed: `{summary['misses_total']}`.",
        f"- Oracle-only-pair misses: `{summary['oracle_only_pair_misses']}`.",
        f"- Beam/sampling oracle score misses: `{summary['beam_sampling_score_misses']}`.",
        f"- Score gaps: `{summary['near_tie_gap_le_0_10']}` near (`<=0.10`), `{summary['moderate_gap_0_10_to_0_25']}` moderate, `{summary['large_gap_gt_0_25']}` large (`>0.25`).",
        f"- Pair-only score gaps: `{summary['pair_near_tie_gap_le_0_10']}` near and `{summary['pair_large_gap_gt_0_25']}` large.",
        f"- Average score gap: `{summary['avg_score_gap']:.6f}` overall, `{summary['avg_pair_only_score_gap']:.6f}` for oracle-only-pair misses.",
        "",
        "The newly introduced paired oracles usually lose because the active+weak role-wise score assigns a lower distance to a non-oracle. This is not mostly a tie-break problem: only two of the twelve misses are within `0.10`, and only one of the seven oracle-only-pair misses is within `0.10`.",
        "",
        "For oracle-only-pair misses, the selected candidate's advantage is dominated by `active_kramers_moyal` in all seven cases. That points to scorer/residual behavior around paired drift/diffusion combinations, not just candidate availability.",
        "",
        "## Aggregate Diagnosis",
        "",
        "| Diagnostic | Count |",
        "| --- | ---: |",
        f"| Selected misses | {summary['misses_total']} |",
        f"| Oracle only appears from pair | {summary['oracle_only_pair_misses']} |",
        f"| Oracle from beam/sampling but score loses | {summary['beam_sampling_score_misses']} |",
        f"| Same diffusion, wrong drift | {summary['miss_side_counts'].get('drift', 0)} |",
        f"| Same drift, wrong diffusion | {summary['miss_side_counts'].get('diffusion', 0)} |",
        f"| Both drift and diffusion wrong | {summary['miss_side_counts'].get('both', 0)} |",
        f"| Oracle has lower model score | {summary['oracle_lower_model_score_count']} |",
        f"| Oracle-only-pair and lower model score | {summary['pair_only_oracle_lower_model_score_count']} |",
        f"| Selected source is pair | {summary['selected_source_counts'].get('pair', 0)} |",
        f"| Selected source is beam | {summary['selected_source_counts'].get('beam', 0)} |",
        "",
        "Model score conflict is not systematic for the oracle-only-pair misses: only one of seven pair-only oracle misses has a lower oracle model score. It is more visible in non-pair score misses, where four of five have a lower oracle model score. Since this eval selected by rerank distance, not by model score except in tie logic, model-score preference alone does not explain the expanded-pairing regression.",
        "",
        "## Per-Sample Miss Table",
        "",
        "| Sample | Selected source | Oracle source | Pair-only oracle | Miss side | Score gap | Dominant selected advantage | Selected model | Oracle model | Same drift | Same diffusion |",
        "| ---: | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for case in cases:
        lines.append(
            "| {sample_index} | {selected_source} | {oracle_source} | {pair_only} | {miss_side} | {gap:.6f} | {dominant} | {selected_model:.6f} | {oracle_model:.6f} | {same_drift} | {same_diffusion} |".format(
                sample_index=case["sample_index"],
                selected_source=case["selected_source"],
                oracle_source=case["oracle_source"],
                pair_only=bool_text(case["oracle_only_pair"]),
                miss_side=case["miss_side"],
                gap=case["score_gap"],
                dominant=case["dominant_selected_advantage"],
                selected_model=case["selected_model_score"],
                oracle_model=case["oracle_model_score"],
                same_drift=bool_text(case["shares_drift"]),
                same_diffusion=bool_text(case["shares_diffusion"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Sample Details",
            "",
        ]
    )
    for case in cases:
        selected_segments = case["selected_segment_distances"]
        oracle_segments = case["oracle_segment_distances"]
        lines.extend(
            [
                f"### Sample {case['sample_index']}",
                "",
                f"- Truth: `{case['truth_sequence']}`",
                f"- Selected: `{case['selected_sequence']}`",
                f"- Best oracle: `{case['oracle_sequence']}`",
                f"- Sources: selected `{case['selected_source']}`, oracle `{case['oracle_source']}`, oracle-only-pair `{bool_text(case['oracle_only_pair'])}`.",
                f"- Scores: selected `{case['selected_score']:.6f}`, oracle `{case['oracle_score']:.6f}`, gap `{case['score_gap']:.6f}`.",
                f"- Model scores: selected `{case['selected_model_score']:.6f}`, oracle `{case['oracle_model_score']:.6f}`, oracle lower `{bool_text(case['oracle_lower_model_score'])}`.",
                f"- Selected drift: `{case['selected_drift']}`",
                f"- Oracle drift: `{case['oracle_drift']}`",
                f"- Selected diffusion: `{case['selected_diffusion']}`",
                f"- Oracle diffusion: `{case['oracle_diffusion']}`",
                f"- Role-wise constants: selected drift `{case['selected_best_drift_constant']}`, selected diffusion `{case['selected_best_diffusion_constant']}`, oracle drift `{case['oracle_best_drift_constant']}`, oracle diffusion `{case['oracle_best_diffusion_constant']}`.",
                f"- Active distances: selected `{selected_segments['active_kramers_moyal']:.6f}`, oracle `{oracle_segments['active_kramers_moyal']:.6f}`.",
                f"- Weak distances: selected `{selected_segments['gaussian_weak_kernel']:.6f}`, oracle `{oracle_segments['gaussian_weak_kernel']:.6f}`.",
                f"- Match flags: same drift `{bool_text(case['shares_drift'])}`, same diffusion `{bool_text(case['shares_diffusion'])}`, miss side `{case['miss_side']}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendation",
            "",
            "Do not make expanded pairing the default and do not run 64-sample expansion from this setting yet.",
            "",
            "A simple pair-aware tie-break is not justified from this log alone. Most paired oracles do not lose by epsilon-sized gaps; they lose because active+weak distance prefers non-oracle candidates. A large pair bonus would be needed for several cases and would likely overfit or promote wrong pair candidates.",
            "",
            "Candidate pruning or pair-aware ranking diagnostics are more justified than a blanket pair bonus. Wrong pair candidates are selected in eight of the twelve misses, and expanded pairing increases both useful and harmful paired candidates. The next low-cost step should inspect the top paired non-oracles and compare their active/weak residuals against the paired oracles, especially for the seven oracle-only-pair misses.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def json_ready(value):
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    text = args.source_log.read_text()
    metrics = parse_metrics(text)
    cases = parse_oracle_misses(text)
    summary = summarize(cases)
    write_report(args.report, metrics, cases, summary)
    payload = {
        "source_log": str(args.source_log),
        "metrics": metrics,
        "summary": json_ready(summary),
        "miss_cases": cases,
    }
    args.json_output.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    print(f"wrote_report={args.report}")
    print(f"wrote_json={args.json_output}")
    print(f"misses_total={summary['misses_total']}")
    print(f"oracle_only_pair_misses={summary['oracle_only_pair_misses']}")
    print(f"near_tie_gap_le_0_10={summary['near_tie_gap_le_0_10']}")


if __name__ == "__main__":
    main()
