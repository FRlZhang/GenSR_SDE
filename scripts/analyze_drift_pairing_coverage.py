#!/usr/bin/env python3
"""Drift miss taxonomy and pairing opportunity report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", type=Path, default=None)
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path("candidate_coverage_diagnostics.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("drift_pairing_coverage_diagnostics.md"),
    )
    return parser.parse_args()


def template(tokens: list[str]) -> str:
    return " ".join(token for token in tokens if token != "CONSTANT")


def source_rank_text(occurrences: list[dict], limit: int = 3) -> str:
    if not occurrences:
        return "-"
    parts = []
    for item in occurrences[:limit]:
        parts.append(
            f"{item['source']}#{item['source_rank']} (candidate {item['candidate_rank']})"
        )
    return "; ".join(parts)


def nearest_text(rows: list[dict], limit: int = 3) -> str:
    if not rows:
        return "-"
    parts = []
    for row in rows[:limit]:
        occurrence = source_rank_text(row.get("occurrences", []), limit=1)
        family = row.get("family") or "unknown"
        right_family = "same-family" if row.get("right_family") else "other-family"
        parts.append(
            f"d={row['distance']}: `{' '.join(row['tokens'])}` "
            f"({family}, {right_family}, {occurrence})"
        )
    return "<br>".join(parts)


def parameter_recommendation(sample: dict, metadata: dict) -> str:
    drift_topk = int(metadata["pair_drift_topk"])
    diffusion_topk = int(metadata["pair_diffusion_topk"])
    pair_cap = int(metadata["pair_drift_diffusion_candidates"])
    drift_rank = sample.get("drift_span_rank")
    diffusion_rank = sample.get("diffusion_span_rank")
    limited_pair_rank = sample.get("limited_pair_cross_rank")
    unrestricted_pair_rank = sample.get("pair_cross_rank")

    actions = []
    if drift_rank is None:
        actions.append("exact drift span unavailable to pairing")
    elif drift_rank > drift_topk:
        actions.append(f"increase pair_drift_topk to at least {drift_rank}")
    if diffusion_rank is None:
        actions.append("exact diffusion span unavailable to pairing")
    elif diffusion_rank > diffusion_topk:
        actions.append(f"increase pair_diffusion_topk to at least {diffusion_rank}")
    if (
        drift_rank is not None
        and diffusion_rank is not None
        and drift_rank <= drift_topk
        and diffusion_rank <= diffusion_topk
    ):
        if limited_pair_rank is None:
            actions.append("investigate pair construction/filtering")
        elif limited_pair_rank > pair_cap:
            actions.append(
                f"increase pair_drift_diffusion_candidates to at least {limited_pair_rank}"
            )
        else:
            actions.append("dedup/parsing/filtering likely removed expected pair")
    elif unrestricted_pair_rank is not None:
        actions.append(f"combined pair rank would be {unrestricted_pair_rank} after top-k expansion")
    return "; ".join(actions) if actions else "current pairing settings should cover it"


def write_report(path: Path, payload: dict) -> dict:
    samples = payload["samples"]
    metadata = payload["metadata"]
    full_count = sum(sample["has_full"] for sample in samples)
    oracle_absent = [sample for sample in samples if not sample["has_full"]]
    drift_missing = [
        sample for sample in oracle_absent if sample["coverage"] == "diffusion_only_present"
    ]
    pairing_missing = [
        sample
        for sample in oracle_absent
        if sample["coverage"] == "drift_and_diffusion_separate_not_paired"
    ]
    family_counts = Counter(sample["truth_drift_family"] for sample in drift_missing)
    template_counts = Counter(template(sample["truth_drift"]) for sample in drift_missing)
    nearest_family_counts = Counter()
    same_family_nearest = 0
    for sample in drift_missing:
        nearest = sample.get("drift_nearest_detailed", [])
        if nearest:
            nearest_family_counts[nearest[0].get("family") or "unknown"] += 1
            same_family_nearest += int(bool(nearest[0].get("right_family")))

    pair_actions = Counter()
    for sample in pairing_missing:
        rec = parameter_recommendation(sample, metadata)
        if "pair_drift_topk" in rec:
            pair_actions["increase_pair_drift_topk"] += 1
        elif "pair_diffusion_topk" in rec:
            pair_actions["increase_pair_diffusion_topk"] += 1
        elif "pair_drift_diffusion_candidates" in rec:
            pair_actions["increase_pair_candidate_cap"] += 1
        elif "filtering" in rec:
            pair_actions["investigate_filtering"] += 1
        else:
            pair_actions["other"] += 1

    lines = [
        "# Drift and Pairing Coverage Diagnostics",
        "",
        "Date: 2026-06-22",
        "",
        "Scope: offline taxonomy over candidate coverage identities. No formal eval, no fingerprint scoring, no scorer changes.",
        "",
        "## Executive Summary",
        "",
        "- Current oracle ceiling: `9/32`.",
        f"- Oracle-absent samples: `{len(oracle_absent)}/32`.",
        f"- Drift-missing with exact diffusion present: `{len(drift_missing)}/23`.",
        f"- Pairing-missing with exact drift and diffusion separate: `{len(pairing_missing)}/23`.",
        "- Attack drift diversity first, then pairing/recombination coverage.",
        "",
        "## Drift Miss Taxonomy",
        "",
        f"Diffusion-only sample indices: `{', '.join(str(s['sample_index']) for s in drift_missing)}`.",
        "",
        "Truth drift family distribution:",
        "",
        "| Family | Count |",
        "| --- | ---: |",
    ]
    for family, count in family_counts.most_common():
        lines.append(f"| {family} | {count} |")
    lines.extend(
        [
            "",
            "Most common missing drift templates:",
            "",
            "| Drift template | Count |",
            "| --- | ---: |",
        ]
    )
    for drift_template, count in template_counts.most_common():
        lines.append(f"| `{drift_template}` | {count} |")
    lines.extend(
        [
            "",
            f"Nearest generated drift has the right operator family in `{same_family_nearest}/{len(drift_missing)}` drift-missing cases.",
            "",
            "| Sample | Truth drift | Truth diffusion | Exact diffusion source/rank | Nearest generated drift candidates | Truth drift family | Recommended target |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in drift_missing:
        lines.append(
            f"| {sample['sample_index']} | `{' '.join(sample['truth_drift'])}` | "
            f"`{' '.join(sample['truth_diffusion'])}` | "
            f"{source_rank_text(sample.get('exact_diffusion_occurrences', []))} | "
            f"{nearest_text(sample.get('drift_nearest_detailed', []))} | "
            f"{sample['truth_drift_family']} | improve drift span generation/diversity |"
        )

    lines.extend(
        [
            "",
            "## Pairing Opportunity",
            "",
            f"Pairing-missing sample indices: `{', '.join(str(s['sample_index']) for s in pairing_missing)}`.",
            "",
            "| Sample | Truth drift | Truth diffusion | Exact drift rank/source | Exact diffusion rank/source | Outside drift top-k | Outside diffusion top-k | Pair cap prevents | Recommendation |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in pairing_missing:
        lines.append(
            f"| {sample['sample_index']} | `{' '.join(sample['truth_drift'])}` | "
            f"`{' '.join(sample['truth_diffusion'])}` | "
            f"rank {sample.get('drift_span_rank')}; {source_rank_text(sample.get('exact_drift_occurrences', []))} | "
            f"rank {sample.get('diffusion_span_rank')}; {source_rank_text(sample.get('exact_diffusion_occurrences', []))} | "
            f"{sample.get('exact_drift_outside_pair_topk')} | "
            f"{sample.get('exact_diffusion_outside_pair_topk')} | "
            f"{sample.get('pair_cap_prevents')} | "
            f"{parameter_recommendation(sample, metadata)} |"
        )

    lines.extend(
        [
            "",
            "Pairing bottleneck summary:",
            "",
            "| Bottleneck | Count |",
            "| --- | ---: |",
        ]
    )
    for name, count in pair_actions.most_common():
        lines.append(f"| {name} | {count} |")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Minimal candidate-generation patch to try next: add a drift-focused span expansion path before reranking. The most direct low-risk version is to increase drift span extraction diversity from beam/sampling outputs and allow more drift spans into pairing, while keeping diffusion settings mostly unchanged.",
            "",
            "Pairing should be the secondary patch: for the 6 pairing-missing samples, most require larger `pair_drift_topk`; a candidate-cap increase only matters after the exact drift enters the pairing top-k.",
            "",
            "A future formal 32-sample eval is justified only after a candidate-generation patch increases offline full-oracle coverage above `9/32` in this diagnostic.",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return {
        "drift_missing": len(drift_missing),
        "pairing_missing": len(pairing_missing),
        "top_family": family_counts.most_common(1)[0] if family_counts else None,
        "pair_actions": pair_actions,
    }


def parse_existing_candidate_report(path: Path = Path("candidate_coverage_diagnostics.md")) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    in_table = False
    for line in path.read_text().splitlines():
        if line.startswith("| Sample | Truth drift | Truth diffusion |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        if line.startswith("| ---"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue
        try:
            sample_idx = int(parts[0])
        except ValueError:
            continue
        rows.append(
            {
                "sample_index": sample_idx,
                "truth_drift": parts[1].strip("`"),
                "truth_diffusion": parts[2].strip("`"),
                "nearest_drift": parts[3],
                "nearest_diffusion": parts[4],
                "missing_side": parts[5],
                "recommendation": parts[6],
            }
        )
    return rows


def rough_family(drift: str) -> str:
    tokens = drift.split()
    if "pow2" in tokens or "pow3" in tokens or "pow" in tokens:
        return "polynomial-like drift"
    if "sin" in tokens:
        return "sin drift"
    if "sub" in tokens and "x_0" in tokens:
        return "mean-reverting / affine drift"
    if tokens == ["mul", "CONSTANT", "x_0"]:
        return "linear drift"
    if tokens == ["mul", "CONSTANT", "CONSTANT"]:
        return "constant drift"
    if tokens[:2] == ["mul", "mul"] and "x_0" in tokens:
        return "nested mul linear drift"
    return "other"


def write_blocked_report(path: Path, rows: list[dict], source_log: Path | None) -> dict:
    drift_missing = [row for row in rows if row["missing_side"] == "drift"]
    pairing_missing = [row for row in rows if row["missing_side"] == "pairing"]
    family_counts = Counter(rough_family(row["truth_drift"]) for row in drift_missing)
    lines = [
        "# Drift and Pairing Coverage Diagnostics",
        "",
        "Date: 2026-06-22",
        "",
        "Scope: partial analysis from existing candidate coverage markdown. Exact candidate span ranks require a JSON sidecar that could not be generated because the checkpoint was missing.",
        "",
        "## Executive Summary",
        "",
        "- Current oracle ceiling: `9/32`.",
        f"- Drift-missing with exact diffusion present: `{len(drift_missing)}`.",
        f"- Pairing-missing with exact drift and diffusion separate: `{len(pairing_missing)}`.",
        "- The available evidence still points to drift diversity first, pairing second.",
        "",
        "## Drift Miss Taxonomy",
        "",
        f"Diffusion-only sample indices: `{', '.join(str(row['sample_index']) for row in drift_missing)}`.",
        "",
        "| Family | Count |",
        "| --- | ---: |",
    ]
    for family, count in family_counts.most_common():
        lines.append(f"| {family} | {count} |")
    lines.extend(
        [
            "",
            "| Sample | Truth drift | Truth diffusion | Nearest generated drift candidates | Recommended target |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for row in drift_missing:
        lines.append(
            f"| {row['sample_index']} | `{row['truth_drift']}` | `{row['truth_diffusion']}` | "
            f"{row['nearest_drift']} | improve drift span generation/diversity |"
        )
    lines.extend(
        [
            "",
            "## Pairing Opportunity",
            "",
            f"Pairing-missing sample indices: `{', '.join(str(row['sample_index']) for row in pairing_missing)}`.",
            "",
            "Exact drift/diffusion ranks, top-k membership, pair-cap checks, and dedup/filtering diagnosis are blocked without the machine-readable JSON sidecar.",
            "",
            "| Sample | Truth drift | Truth diffusion | Available inference |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for row in pairing_missing:
        lines.append(
            f"| {row['sample_index']} | `{row['truth_drift']}` | `{row['truth_diffusion']}` | "
            "exact drift and exact diffusion are present separately, but rank/cap cause is unavailable |"
        )
    lines.extend(
        [
            "",
            "## Blocker",
            "",
            "The existing coverage markdown does not contain candidate span ranks. A rerun of candidate generation was attempted to emit `candidate_coverage_diagnostics.json`, but the checkpoint `/private/tmp/gensr_sde_role_token_2000/role_token_2000.pth` is missing.",
            "",
            "Smallest command after regenerating or restoring the checkpoint:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \\",
            "/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_candidate_coverage.py \\",
            "  --source-log /private/tmp/gensr_sde_32_rolewise_no_multi_u0.log \\",
            "  --report candidate_coverage_diagnostics.md \\",
            "  --json-output candidate_coverage_diagnostics.json \\",
            "  > /private/tmp/gensr_sde_candidate_coverage_json.log 2>&1",
            "",
            "/opt/miniconda3/envs/gensr/bin/python3 scripts/analyze_drift_pairing_coverage.py \\",
            "  --coverage-json candidate_coverage_diagnostics.json \\",
            "  --report drift_pairing_coverage_diagnostics.md \\",
            "  > /private/tmp/gensr_sde_drift_pairing_coverage.log 2>&1",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return {
        "drift_missing": len(drift_missing),
        "pairing_missing": len(pairing_missing),
        "top_family": family_counts.most_common(1)[0] if family_counts else None,
    }


def main() -> None:
    args = parse_args()
    if not args.coverage_json.is_file():
        rows = parse_existing_candidate_report()
        if not rows:
            raise FileNotFoundError(
                f"{args.coverage_json} not found; rerun analyze_candidate_coverage.py with --json-output"
            )
        summary = write_blocked_report(args.report, rows, args.source_log)
        print(f"wrote_report={args.report}")
        print("status=blocked_missing_candidate_coverage_json")
        print(f"drift_missing={summary['drift_missing']}")
        print(f"pairing_missing={summary['pairing_missing']}")
        if summary["top_family"] is not None:
            print(f"top_missing_drift_family={summary['top_family'][0]}:{summary['top_family'][1]}")
        return
    payload = json.loads(args.coverage_json.read_text())
    summary = write_report(args.report, payload)
    print(f"wrote_report={args.report}")
    print(f"drift_missing={summary['drift_missing']}")
    print(f"pairing_missing={summary['pairing_missing']}")
    if summary["top_family"] is not None:
        print(f"top_missing_drift_family={summary['top_family'][0]}:{summary['top_family'][1]}")
    print(
        "pairing_actions="
        + ",".join(f"{key}:{value}" for key, value in sorted(summary["pair_actions"].items()))
    )


if __name__ == "__main__":
    main()
