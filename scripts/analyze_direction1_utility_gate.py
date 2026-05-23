#!/usr/bin/env python3
"""Build Direction 1 utility-gate audit tables and report.

This script is intentionally post-hoc: it reads completed experiment logs and
prediction traces, then writes tables for calibration/test sanity checks. It
does not choose deployable thresholds from test labels.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


RAW_BASELINE = {
    "a_avg": 88.4417,
    "a_last": 82.91,
    "compact_mb": 7.6598,
    "actual_mb": 22.7962,
}


METHOD_LABELS = {
    "cub_direction1_fallback_candidate_rule_count_r3": "candidate-only rule gate",
    "cub_direction1_fallback_delta_ridge_count_r3": "delta regression ridge",
    "cub_direction1_fallback_benefit_harm_count_r3": "benefit-vs-harm logistic",
    "cub_direction1_fallback_delta_ridge_pair_count_r3": "utility + pair/node shrinkage",
    "cub_direction1_fallback_delta_ridge_class_pair_count_r3": "class_pair_only",
    "cub_direction1_fallback_delta_ridge_node_pair_count_r3": "node_pair_only",
}


TRACE_DIRS = {
    "candidate-only rule gate": "logs/direction1_utility_traces/candidate_rule",
    "delta regression ridge": "logs/direction1_utility_traces/delta_ridge",
    "benefit-vs-harm logistic": "logs/direction1_utility_traces/benefit_harm",
    "utility + pair/node shrinkage": "logs/direction1_utility_traces/delta_ridge_pair",
    "class_pair_only": "logs/direction1_utility_traces/class_pair",
    "node_pair_only": "logs/direction1_utility_traces/node_pair",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_csv",
        default="logs/direction1_utility_runs/utility_gate_results_raw.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/direction1_utility_runs",
    )
    parser.add_argument(
        "--report",
        default="logs/direction1_utility_gate_report.md",
    )
    parser.add_argument(
        "--reference_csv",
        default="logs/direction1_utility_runs/reference_controls_raw.csv",
    )
    parser.add_argument(
        "--random_low_budget_csv",
        default="logs/direction1_utility_runs/random_low_budget_controls_stdout.csv",
    )
    return parser.parse_args()


def pct(x) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return float(x) * 100.0


def pp(x) -> float:
    return pct(x)


def fmt(value, digits=2, suffix=""):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def read_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = df["prefix"].map(METHOD_LABELS).fillna(df["prefix"])
    return df


def final_trace_file(trace_dir: Path) -> Path:
    paths = sorted(trace_dir.glob("*task19*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No final-task trace found in {trace_dir}")
    return paths[-1]


def read_trace(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty trace file: {path}")
    df = pd.DataFrame(rows)
    for col in [
        "compact_correct",
        "fallback_correct",
        "final_correct",
        "fallback_available",
        "fallback_used",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    numeric_cols = [
        "compact_top1",
        "raw_top1_class",
        "fallback_top1",
        "compact_margin",
        "raw_margin",
        "selected_node_residual",
        "score_cross_gap",
        "compact_rank_of_raw_top1",
        "raw_rank_of_compact_top1",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def quadrant_from_rates(n, compact_acc, fallback_acc, benefit_rate):
    benefit = int(round(n * benefit_rate))
    harm_rate = compact_acc - fallback_acc + benefit_rate
    harm = int(round(n * harm_rate))
    both_correct = int(round(n * (fallback_acc - benefit_rate)))
    both_wrong = n - benefit - harm - both_correct
    return {
        "split": "calibration",
        "n": n,
        "both_correct": both_correct,
        "benefit": benefit,
        "harm": harm,
        "both_wrong": both_wrong,
        "compact_acc": compact_acc,
        "raw_fallback_acc": fallback_acc,
        "benefit_rate": benefit / n,
        "harm_rate": harm / n,
        "neutral_rate": (both_correct + both_wrong) / n,
        "oracle_gain": benefit / n,
    }


def quadrant_from_trace(df: pd.DataFrame, split: str):
    compact = df["compact_correct"].astype(bool)
    raw = df["fallback_correct"].astype(bool)
    both_correct = int((compact & raw).sum())
    benefit = int((~compact & raw).sum())
    harm = int((compact & ~raw).sum())
    both_wrong = int((~compact & ~raw).sum())
    n = int(len(df))
    return {
        "split": split,
        "n": n,
        "both_correct": both_correct,
        "benefit": benefit,
        "harm": harm,
        "both_wrong": both_wrong,
        "compact_acc": float(compact.mean()),
        "raw_fallback_acc": float(raw.mean()),
        "benefit_rate": benefit / n,
        "harm_rate": harm / n,
        "neutral_rate": (both_correct + both_wrong) / n,
        "oracle_gain": benefit / n,
    }


def selected_stats_from_trace(df: pd.DataFrame):
    compact = df["compact_correct"].astype(bool)
    raw = df["fallback_correct"].astype(bool)
    used = df["fallback_used"].astype(bool)
    benefit = used & ~compact & raw
    harm = used & compact & ~raw
    neutral = used & (compact == raw)
    n = len(df)
    b = int(benefit.sum())
    h = int(harm.sum())
    u = int(used.sum())
    return {
        "fallback_used_count": u,
        "fallback_rate_trace": u / n,
        "benefit_selected_trace": b,
        "harm_selected_trace": h,
        "neutral_selected_trace": int(neutral.sum()),
        "test_net_gain_trace": (b - h) / n,
        "utility_precision_trace": b / (b + h) if (b + h) else 0.0,
    }


def rule_summary(df: pd.DataFrame, name: str, mask: pd.Series):
    n = len(df)
    compact = df["compact_correct"].astype(bool)
    raw = df["fallback_correct"].astype(bool)
    mask = mask.fillna(False).astype(bool)
    m = int(mask.sum())
    benefit = int((mask & ~compact & raw).sum())
    harm = int((mask & compact & ~raw).sum())
    neutral = m - benefit - harm
    return {
        "rule": name,
        "n": n,
        "candidate_count": m,
        "candidate_rate": m / n,
        "benefit_count": benefit,
        "harm_count": harm,
        "neutral_count": neutral,
        "benefit_rate_in_candidate": benefit / m if m else 0.0,
        "harm_rate_in_candidate": harm / m if m else 0.0,
        "neutral_rate_in_candidate": neutral / m if m else 0.0,
        "net_gain_overall": (benefit - harm) / n,
        "net_gain_in_candidate": (benefit - harm) / m if m else 0.0,
    }


def candidate_rules(df: pd.DataFrame) -> List[Dict[str, object]]:
    raw_top1 = df["raw_top1_class"] if "raw_top1_class" in df else df["fallback_top1"]
    disagree = df["compact_top1"] != raw_top1
    q30_compact_margin = df["compact_margin"].quantile(0.30)
    q70_raw_margin = df["raw_margin"].quantile(0.70)
    q50_residual = df["selected_node_residual"].quantile(0.50)
    q70_cross_gap = df["score_cross_gap"].quantile(0.70)
    rank_rule = (
        (df["compact_rank_of_raw_top1"] <= 3)
        & (df["raw_rank_of_compact_top1"] > 1)
    )
    margin_raw = disagree & (df["compact_margin"] < q30_compact_margin) & (df["raw_margin"] > q70_raw_margin)
    residual_cross = disagree & (df["selected_node_residual"] > q50_residual) & (df["score_cross_gap"] > q70_cross_gap)
    enhanced = disagree & rank_rule & (df["selected_node_residual"] > q50_residual) & (df["score_cross_gap"] > q70_cross_gap)
    return [
        rule_summary(df, "all_available", pd.Series(True, index=df.index)),
        rule_summary(df, "disagree", disagree),
        rule_summary(df, "margin_raw_q30_q70", margin_raw),
        rule_summary(df, "rank_rule", disagree & rank_rule),
        rule_summary(df, "residual_cross_gap", residual_cross),
        rule_summary(df, "enhanced", enhanced),
    ]


def add_derived_result_columns(df: pd.DataFrame, selected_baseline: pd.Series):
    out = df.copy()
    selected_a_avg = float(selected_baseline["avg_accuracy"])
    selected_a_last = float(selected_baseline["final_top1"])
    out["selected_relative_a_last_pp"] = out["final_top1"] - selected_a_last
    out["selected_relative_a_avg_pp"] = out["avg_accuracy"] - selected_a_avg
    out["raw_relative_a_last_pp"] = out["final_top1"] - RAW_BASELINE["a_last"]
    out["compact_memory_reduction_vs_raw"] = 1.0 - (out["memory_compact_mb"] / RAW_BASELINE["compact_mb"])
    out["actual_memory_reduction_vs_raw"] = 1.0 - (out["memory_actual_mb"] / RAW_BASELINE["actual_mb"])
    old = pd.to_numeric(out["acc_old"], errors="coerce")
    new = pd.to_numeric(out["acc_new"], errors="coerce")
    out["old_new_hm"] = np.where((old + new) > 0, 2 * old * new / (old + new), np.nan)
    return out


def label_reference_row(row: pd.Series) -> str:
    log_file = str(row.get("log_file", ""))
    prefix = str(row.get("prefix", ""))
    if "rate001" in log_file:
        return "random fallback 1% control"
    if "rate002" in log_file:
        return "random fallback 2% control"
    if "static" in prefix:
        return "static fallback count-r3"
    if "random" in prefix:
        return "random fallback 10% control"
    if "oracle" in prefix:
        return "oracle count-r3"
    if "ridge" in prefix:
        return "current ridge gate"
    if "gbdt" in prefix and "max1" in prefix:
        return "GBDT diagnostic max1"
    if "gbdt" in prefix and "max2" in prefix:
        return "GBDT diagnostic max2"
    return prefix or log_file


def read_optional_reference_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "final_top1" not in df.columns:
        return pd.DataFrame()
    df = df[pd.to_numeric(df["final_top1"], errors="coerce").notna()].copy()
    if df.empty:
        return df
    df["method"] = df.apply(label_reference_row, axis=1)
    for col in ["raw_fallback_compact_acc", "raw_fallback_final_acc"]:
        if col not in df.columns:
            df[col] = np.nan
    df["test_net_pp"] = (
        pd.to_numeric(df["raw_fallback_final_acc"], errors="coerce")
        - pd.to_numeric(df["raw_fallback_compact_acc"], errors="coerce")
    ) * 100.0
    return df


def write_csv(path: Path, rows: Iterable[Dict[str, object]]):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_report(
    report_path: Path,
    results: pd.DataFrame,
    quadrants: List[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    reference_controls: pd.DataFrame,
):
    selected = results[results["method"] == "candidate-only rule gate"].iloc[0]
    full_pair = results[results["method"] == "utility + pair/node shrinkage"].iloc[0]
    selected_last = float(selected["final_top1"])
    selected_avg = float(selected["avg_accuracy"])

    q_rows = []
    for row in quadrants:
        q_rows.append([
            row["split"],
            row["n"],
            f"{row['both_correct']} ({pct(row['both_correct'] / row['n']):.2f}%)",
            f"{row['benefit']} ({pct(row['benefit_rate']):.2f}%)",
            f"{row['harm']} ({pct(row['harm_rate']):.2f}%)",
            f"{row['both_wrong']} ({pct(row['both_wrong'] / row['n']):.2f}%)",
            fmt(pct(row["compact_acc"]), 2, "%"),
            fmt(pct(row["raw_fallback_acc"]), 2, "%"),
            fmt(pp(row["oracle_gain"]), 2, "pp"),
        ])

    rule_rows = []
    for row in candidate_rows:
        rule_rows.append([
            row["rule"],
            f"{row['candidate_count']} / {row['n']}",
            fmt(pct(row["candidate_rate"]), 2, "%"),
            fmt(pct(row["benefit_rate_in_candidate"]), 2, "%"),
            fmt(pct(row["harm_rate_in_candidate"]), 2, "%"),
            fmt(pp(row["net_gain_overall"]), 2, "pp"),
        ])

    result_rows = []
    for _, row in results.iterrows():
        result_rows.append([
            row["method"],
            fmt(row["avg_accuracy"], 4),
            fmt(row["final_top1"], 2),
            fmt(row["selected_relative_a_last_pp"], 2, "pp"),
            fmt(row["selected_relative_a_avg_pp"], 2, "pp"),
            fmt(row["raw_relative_a_last_pp"], 2, "pp"),
            fmt(row["old_new_hm"], 2),
            fmt(pct(row["raw_fallback_rate"]), 2, "%"),
            int(row["raw_fallback_benefit_sel"]),
            int(row["raw_fallback_harm_sel"]),
            fmt(pp(row["raw_fallback_net"]), 2, "pp"),
            fmt(pct(row["raw_fallback_gate_fallback_rate"]), 2, "%"),
            fmt(pp(row["raw_fallback_gate_net"]), 2, "pp"),
            fmt(row["memory_compact_mb"], 4),
            fmt(row["memory_actual_mb"], 4),
        ])

    reference_rows = []
    if reference_controls is not None and not reference_controls.empty:
        preferred_order = [
            "static fallback count-r3",
            "random fallback 10% control",
            "random fallback 1% control",
            "random fallback 2% control",
            "current ridge gate",
            "GBDT diagnostic max1",
            "GBDT diagnostic max2",
            "oracle count-r3",
        ]
        refs = reference_controls.copy()
        refs["_order"] = refs["method"].apply(
            lambda x: preferred_order.index(x) if x in preferred_order else 999
        )
        refs = refs.sort_values(["_order", "method"])
        for _, row in refs.iterrows():
            reference_rows.append([
                row["method"],
                fmt(row["avg_accuracy"], 4),
                fmt(row["final_top1"], 2),
                fmt(pct(row.get("raw_fallback_rate", np.nan)), 2, "%"),
                fmt(pct(row.get("raw_fallback_compact_acc", np.nan)), 2, "%"),
                fmt(pct(row.get("raw_fallback_final_acc", np.nan)), 2, "%"),
                fmt(row.get("test_net_pp", np.nan), 2, "pp"),
                fmt(pct(row.get("raw_fallback_oracle_improvable", np.nan)), 2, "%"),
                fmt(row.get("memory_compact_mb", np.nan), 4),
                fmt(row.get("memory_actual_mb", np.nan), 4),
            ])

    pair_rows = []
    for method in ["delta regression ridge", "class_pair_only", "node_pair_only", "utility + pair/node shrinkage"]:
        row = results[results["method"] == method].iloc[0]
        pair_rows.append([
            method,
            fmt(row["final_top1"], 2),
            fmt(row["selected_relative_a_last_pp"], 2, "pp"),
            fmt(pp(row["raw_fallback_net"]), 2, "pp"),
            fmt(pct(row["raw_fallback_rate"]), 2, "%"),
            int(row["raw_fallback_benefit_sel"]),
            int(row["raw_fallback_harm_sel"]),
            fmt(pct(row["raw_fallback_precision"]), 2, "%"),
            fmt(row["memory_fallback_gate_mb"], 4),
        ])

    memory_rows = []
    for method in ["candidate-only rule gate", "utility + pair/node shrinkage"]:
        row = results[results["method"] == method].iloc[0]
        memory_rows.append([
            method,
            fmt(row["memory_compact_mb"], 4),
            fmt(row["memory_actual_mb"], 4),
            fmt(row["memory_fallback_mb"], 4),
            fmt(row["memory_fallback_gate_mb"], 4),
            fmt(pct(row["compact_memory_reduction_vs_raw"]), 2, "%"),
            fmt(pct(row["actual_memory_reduction_vs_raw"]), 2, "%"),
        ])

    lines = [
        "# Direction 1 Utility Gate Report",
        "",
        "Scope: CUB-200, seed 1993, cached frozen ViT-B/16 features, classifier-only CPU path. "
        "Growth and additive edge-aware scoring remain disabled. Thresholds/rates are selected on calibration only; "
        "test labels are used here only for post-hoc evaluation.",
        "",
        "## Executive conclusion",
        "",
        "The utility-learning direction is partially validated as a diagnostic but not strong enough for a main paper claim. "
        f"The best deployable variant is `utility + pair/node shrinkage`, with A_Last {fmt(full_pair['final_top1'], 2)} "
        f"versus the same-run compact baseline {fmt(selected_last, 2)}, a {fmt(full_pair['selected_relative_a_last_pp'], 2, 'pp')} gain. "
        f"Its test net gain is {fmt(pp(full_pair['raw_fallback_net']), 2, 'pp')}, but A_Avg drops "
        f"{fmt(abs(full_pair['selected_relative_a_avg_pp']), 2, 'pp')} from the compact selected baseline. "
        "This misses the minimum success rule of selected-relative A_Last >= +0.2pp and A_Avg drop <= 0.1pp.",
        "",
        "The core failure mode is still calibration-to-test transfer: calibration precision is high, but test benefit/harm separation weakens sharply. "
        "The result should be kept as a negative/diagnostic Direction 1 outcome, while the oracle count-r3 upper bound remains useful evidence that compact errors are recoverable in principle.",
        "",
        "## Trace Fields",
        "",
        "The current trace includes the required compact/fallback correctness fields plus the new deployable signals:",
        "",
        "- raw-side confidence: `raw_top1_score`, `raw_top2_score`, `raw_margin`, `raw_distance_to_top1`, `raw_distance_to_top2`",
        "- cross-rank/disagreement: `raw_top1_class`, `compact_rank_of_raw_top1`, `raw_rank_of_compact_top1`, `compact_score_on_raw_top1`, `raw_score_on_compact_top1`, `score_cross_gap`",
        "- node/class context: `selected_class`, `selected_node_id` via `nearest_compact_node`, `task_relation`, `selected_class_age`, `raw_class_age`, `fallback_available`",
        "",
        "## Four-Quadrant Statistics",
        "",
        markdown_table(
            [
                "Split",
                "N",
                "compact correct / raw correct",
                "compact wrong / raw correct",
                "compact correct / raw wrong",
                "compact wrong / raw wrong",
                "compact acc",
                "raw fallback acc",
                "oracle gain",
            ],
            q_rows,
        ),
        "",
        "Interpretation: benefit is much smaller than harm on both calibration and test. Any deployable gate must therefore optimize net utility, not raw correctness.",
        "",
        "## Candidate Rule Search",
        "",
        markdown_table(
            ["Rule", "Candidates", "Candidate rate", "Benefit within", "Harm within", "Overall net gain"],
            rule_rows,
        ),
        "",
        "The simple `disagree` pool raises benefit density relative to all samples but still has negative net gain. "
        "The stricter fixed rules reduce candidate count but do not produce a reliable positive-net candidate pool on test.",
        "",
        "## Reference Controls",
        "",
        markdown_table(
            [
                "Method",
                "A_Avg",
                "A_Last",
                "fallback rate",
                "compact acc",
                "final acc",
                "test net",
                "oracle-improvable",
                "compact MB",
                "actual MB",
            ],
            reference_rows,
        )
        if reference_rows
        else "Reference controls were not found.",
        "",
        "Notes: the 1% and 2% random controls are extra low-budget sanity checks run from the non-calibration random config; "
        "they are useful directionally but not a strict same-topology comparator for the learned utility gates. "
        "The strict utility-gate comparator remains the candidate-only compact baseline in the next table.",
        "",
        "## Utility Gate Results",
        "",
        markdown_table(
            [
                "Method",
                "A_Avg",
                "A_Last",
                "selected-rel A_Last",
                "selected-rel A_Avg",
                "raw-rel A_Last",
                "Old-New HM",
                "test fallback",
                "benefit sel",
                "harm sel",
                "test net",
                "calib fallback",
                "calib net",
                "compact MB",
                "actual MB",
            ],
            result_rows,
        ),
        "",
        f"Selected-relative uses the same calibration setting compact baseline: A_Avg {fmt(selected_avg, 4)}, A_Last {fmt(selected_last, 2)}. "
        f"Raw-relative uses same-feature raw HC-SOINN A_Last {fmt(RAW_BASELINE['a_last'], 2)}.",
        "",
        "## Pair/Node Shrinkage Ablation",
        "",
        markdown_table(
            [
                "Variant",
                "A_Last",
                "selected-rel A_Last",
                "test net",
                "fallback rate",
                "benefit sel",
                "harm sel",
                "utility precision",
                "gate MB",
            ],
            pair_rows,
        ),
        "",
        "`full` is the only tested utility-table variant with positive test net gain. `class_pair_only` is nearly neutral, while `node_pair_only` over-triggers and harms accuracy.",
        "",
        "## Memory Ledger Audit",
        "",
        markdown_table(
            [
                "Method",
                "compact MB",
                "actual MB",
                "raw fallback MB",
                "gate/table MB",
                "compact reduction vs raw",
                "actual reduction vs raw",
            ],
            memory_rows,
        ),
        "",
        f"Raw fallback cache is counted in compact deployable memory. Even with count-r3 fallback and the pair/node table, compact memory remains below raw HC-SOINN by "
        f"{fmt(pct(full_pair['compact_memory_reduction_vs_raw']), 2, '%')}.",
        "",
        "## Success Check",
        "",
        markdown_table(
            ["Criterion", "Status", "Evidence"],
            [
                [
                    "net_gain > 0",
                    "partial pass",
                    f"full pair/node has test net {fmt(pp(full_pair['raw_fallback_net']), 2, 'pp')}; most other deployable gates are <= 0",
                ],
                [
                    "selected-relative A_Last >= +0.2pp",
                    "fail",
                    f"best is {fmt(full_pair['selected_relative_a_last_pp'], 2, 'pp')}",
                ],
                [
                    "A_Avg drop <= 0.1pp",
                    "fail",
                    f"full pair/node A_Avg delta is {fmt(full_pair['selected_relative_a_avg_pp'], 2, 'pp')}",
                ],
                [
                    "raw-relative A_Last >= 0",
                    "fail",
                    f"full pair/node is {fmt(full_pair['raw_relative_a_last_pp'], 2, 'pp')} vs same-feature raw HC-SOINN",
                ],
                [
                    "compact memory reduction vs raw >= 30%",
                    "pass",
                    f"{fmt(pct(full_pair['compact_memory_reduction_vs_raw']), 2, '%')} reduction",
                ],
            ],
        ),
        "",
        "## Decision",
        "",
        "Do not promote the deployable reliability gate to the paper's main claim yet. The correct claim is narrower: "
        "oracle raw-node fallback proves recoverable compact-topology errors, and the utility/pair objective gives the first positive deployable net-gain point, "
        "but its effect is too small and too unstable for the planned success criteria. The next branch should be Direction 2 atom-conflict gate, or a deeper calibration distribution-shift study before spending on 3-seed Direction 1 runs.",
        "",
        "## Artifacts",
        "",
        "- `logs/direction1_utility_runs/utility_gate_results_raw.csv`",
        "- `logs/direction1_utility_runs/utility_gate_results_derived.csv`",
        "- `logs/direction1_utility_runs/four_quadrant_stats.csv`",
        "- `logs/direction1_utility_runs/candidate_rule_search.csv`",
        "- `logs/direction1_utility_runs/pair_node_ablation.csv`",
        "- `logs/direction1_utility_runs/reference_controls_raw.csv`",
        "- `logs/direction1_utility_runs/random_low_budget_controls_stdout.csv`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = read_rows(Path(args.results_csv))
    selected = results[results["method"] == "candidate-only rule gate"]
    if selected.empty:
        raise RuntimeError("candidate-only rule gate row is required as compact baseline")
    traces: Dict[str, pd.DataFrame] = {}
    for method, trace_dir in TRACE_DIRS.items():
        path = final_trace_file(Path(trace_dir))
        traces[method] = read_trace(path)

    selected_stats = []
    for _, row in results.iterrows():
        stats = selected_stats_from_trace(traces[row["method"]])
        for key, value in stats.items():
            results.loc[row.name, key] = value

    results = add_derived_result_columns(results, selected.iloc[0])
    results.to_csv(output_dir / "utility_gate_results_derived.csv", index=False)

    calib_source = selected.iloc[0]
    calib_quad = quadrant_from_rates(
        int(calib_source["raw_fallback_gate_samples"]),
        float(calib_source["raw_fallback_gate_compact_acc"]),
        float(calib_source["raw_fallback_gate_fallback_acc"]),
        float(calib_source["raw_fallback_gate_pos_rate"]),
    )
    test_quad = quadrant_from_trace(traces["candidate-only rule gate"], "test")
    quadrants = [calib_quad, test_quad]
    write_csv(output_dir / "four_quadrant_stats.csv", quadrants)

    rule_rows = candidate_rules(traces["candidate-only rule gate"])
    write_csv(output_dir / "candidate_rule_search.csv", rule_rows)

    reference_controls = read_optional_reference_csv(Path(args.reference_csv))
    random_low_budget = read_optional_reference_csv(Path(args.random_low_budget_csv))
    if not random_low_budget.empty:
        reference_controls = pd.concat([reference_controls, random_low_budget], ignore_index=True)
    if not reference_controls.empty:
        reference_controls.to_csv(output_dir / "reference_controls_derived.csv", index=False)

    pair_methods = [
        "delta regression ridge",
        "class_pair_only",
        "node_pair_only",
        "utility + pair/node shrinkage",
    ]
    pair_rows = results[results["method"].isin(pair_methods)].copy()
    pair_rows.to_csv(output_dir / "pair_node_ablation.csv", index=False)

    build_report(Path(args.report), results, quadrants, rule_rows, reference_controls)

    print(f"Wrote {output_dir / 'utility_gate_results_derived.csv'}")
    print(f"Wrote {output_dir / 'four_quadrant_stats.csv'}")
    print(f"Wrote {output_dir / 'candidate_rule_search.csv'}")
    print(f"Wrote {output_dir / 'pair_node_ablation.csv'}")
    if not reference_controls.empty:
        print(f"Wrote {output_dir / 'reference_controls_derived.csv'}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
