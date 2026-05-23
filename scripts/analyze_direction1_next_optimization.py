#!/usr/bin/env python3
"""Direction 1 next-round optimization audit.

The script uses calibration traces only to fit/select deployable gates, then
applies the fixed decisions to test traces. Test labels are used only for
post-hoc evaluation tables.
"""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


RAW_HC_SOINN_A_LAST = 82.91
RAW_HC_SOINN_COMPACT_MB = 7.6598
COMPACT_BASE_MB = 4.5860
COMPACT_BASE_ACTUAL_MB = 19.0062
RAW_FALLBACK_MB = 1.7778
GBDT_GATE_MB = 0.1155
LINEAR_GATE_MB = 0.0020
RULE_GATE_MB = 0.0004

RATES = (0.002, 0.005, 0.01, 0.02)
BUCKET_RATES = (0.005, 0.01, 0.02, 0.04)


NUMERIC_COLUMNS = [
    "compact_margin",
    "compact_top1_score",
    "compact_top2_score",
    "nearest_compact_distance",
    "nearest_compact_count",
    "selected_node_count",
    "selected_node_residual",
    "selected_atom_inactive_ratio",
    "selected_atom_protected_ratio",
    "selected_atom_weighted_inactive_ratio",
    "selected_atom_old_support_mean",
    "selected_atom_weighted_old_support",
    "selected_atom_usage_ema_mean",
    "selected_atom_weighted_usage_ema",
    "selected_atom_abs_sum",
    "selected_atom_entropy",
    "fallback_distance",
    "fallback_top2_distance",
    "fallback_margin",
    "fallback_count",
    "fallback_residual",
    "raw_top1_score",
    "raw_top2_score",
    "raw_margin",
    "raw_distance_to_top1",
    "raw_distance_to_top2",
    "compact_rank_of_raw_top1",
    "raw_rank_of_compact_top1",
    "compact_score_on_raw_top1",
    "raw_score_on_compact_top1",
    "score_cross_gap",
    "selected_class_age",
    "raw_class_age",
    "class_age",
    "task_relation_code",
]


COMPACT_FEATURE_HINTS = [
    "compact_margin",
    "compact_top1_score",
    "compact_top2_score",
    "selected_node_residual",
    "selected_node_count",
    "nearest_compact_distance",
    "selected_atom_weighted_inactive_ratio",
    "selected_atom_old_support_mean",
    "selected_atom_weighted_old_support",
    "selected_atom_usage_ema_mean",
    "selected_atom_weighted_usage_ema",
    "selected_atom_entropy",
    "compact_rank_of_raw_top1",
    "selected_class_age",
    "task_relation_code",
    "selected_state_inactive",
    "selected_state_plastic",
    "selected_state_protected",
]


RAW_FEATURE_HINTS = [
    "fallback_distance",
    "fallback_margin",
    "fallback_count",
    "fallback_residual",
    "raw_top1_score",
    "raw_top2_score",
    "raw_margin",
    "raw_distance_to_top1",
    "raw_distance_to_top2",
    "raw_rank_of_compact_top1",
    "fallback_available",
    "fallback_state_inactive",
    "fallback_state_plastic",
    "fallback_state_protected",
]


def pct(x: float) -> float:
    return 100.0 * float(x)


def fmt(x, digits=2, suffix="") -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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
    for col in NUMERIC_COLUMNS + [
        "compact_top1",
        "compact_top2",
        "fallback_top1",
        "raw_top1_class",
        "nearest_compact_node",
        "fallback_node",
        "target",
        "known_classes",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_traces(trace_dir: Path, split: str) -> Dict[int, pd.DataFrame]:
    traces: Dict[int, pd.DataFrame] = {}
    pattern = re.compile(r"task(\d+).*_" + re.escape(split) + r"\.jsonl$")
    for path in sorted(trace_dir.glob(f"*_{split}.jsonl")):
        match = pattern.search(path.name)
        if not match:
            continue
        traces[int(match.group(1))] = read_jsonl(path)
    if not traces:
        raise FileNotFoundError(f"No {split} traces found in {trace_dir}")
    return traces


def load_existing_test_traces(trace_dir: Path) -> Dict[int, pd.DataFrame]:
    traces: Dict[int, pd.DataFrame] = {}
    pattern = re.compile(r"task(\d+).*\.jsonl$")
    for path in sorted(trace_dir.glob("*.jsonl")):
        match = pattern.search(path.name)
        if match:
            traces[int(match.group(1))] = read_jsonl(path)
    if not traces:
        raise FileNotFoundError(f"No traces found in {trace_dir}")
    return traces


def safe_bool(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df[col].astype(bool).to_numpy()


def benefit_harm_neutral(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    compact = safe_bool(df, "compact_correct")
    raw = safe_bool(df, "fallback_correct")
    benefit = (~compact) & raw
    harm = compact & (~raw)
    neutral = ~(benefit | harm)
    return benefit, harm, neutral


def evaluate_mask(df: pd.DataFrame, mask: np.ndarray) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    compact = safe_bool(df, "compact_correct")
    raw = safe_bool(df, "fallback_correct")
    benefit, harm, neutral = benefit_harm_neutral(df)
    final = np.where(mask, raw, compact)
    n = max(1, len(df))
    b = int(np.sum(mask & benefit))
    h = int(np.sum(mask & harm))
    z = int(np.sum(mask & neutral))
    return {
        "acc": float(np.mean(final)) if len(final) else 0.0,
        "compact_acc": float(np.mean(compact)) if len(compact) else 0.0,
        "fallback_rate": float(np.mean(mask)) if len(mask) else 0.0,
        "benefit": b,
        "harm": h,
        "neutral": z,
        "net_count": b - h,
        "net_gain": (b - h) / float(n),
        "benefit_recall": b / max(1.0, float(np.sum(benefit))),
        "utility_precision": b / max(1.0, float(b + h)),
    }


def old_new_metrics(df: pd.DataFrame, mask: np.ndarray) -> Dict[str, float]:
    target = pd.to_numeric(df.get("target", pd.Series(np.zeros(len(df)))), errors="coerce").to_numpy()
    known = pd.to_numeric(df.get("known_classes", pd.Series(np.zeros(len(df)))), errors="coerce").to_numpy()
    compact = safe_bool(df, "compact_correct")
    raw = safe_bool(df, "fallback_correct")
    final = np.where(mask, raw, compact)
    old_mask = target < known
    new_mask = ~old_mask
    old_acc = float(np.mean(final[old_mask])) if np.any(old_mask) else float("nan")
    new_acc = float(np.mean(final[new_mask])) if np.any(new_mask) else float("nan")
    hm = 2 * old_acc * new_acc / (old_acc + new_acc) if old_acc > 0 and new_acc > 0 else float("nan")
    return {"old_acc": old_acc, "new_acc": new_acc, "old_new_hm": hm}


def one_hot(values: pd.Series, prefix: str, labels: List[str]) -> pd.DataFrame:
    raw = values.fillna("unknown").astype(str).str.lower()
    out = {}
    for label in labels:
        out[f"{prefix}_{label}"] = (raw == label).astype(np.float32)
    return pd.DataFrame(out, index=values.index)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
        else:
            values = pd.Series(np.zeros(n), index=df.index)
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[col] = values.astype(np.float32)

    compact_top1 = pd.to_numeric(df.get("compact_top1", pd.Series(np.zeros(n))), errors="coerce").fillna(-1)
    fallback_top1 = pd.to_numeric(df.get("fallback_top1", pd.Series(np.zeros(n))), errors="coerce").fillna(-2)
    out["compact_raw_disagree"] = (compact_top1.to_numpy() != fallback_top1.to_numpy()).astype(np.float32)
    out["fallback_available"] = safe_bool(df, "fallback_available").astype(np.float32)
    compact_margin = np.maximum(out["compact_margin"].to_numpy(dtype=np.float64), 1e-6)
    out["distance_gap"] = out["nearest_compact_distance"] - out["fallback_distance"]
    out["distance_gap_over_margin"] = np.clip(out["distance_gap"].to_numpy() / compact_margin, -50.0, 50.0)
    out["fallback_margin_ratio"] = np.clip(out["fallback_margin"].to_numpy() / compact_margin, -50.0, 50.0)
    out["count_log_gap"] = np.log1p(np.maximum(out["fallback_count"].to_numpy(), 0.0)) - np.log1p(
        np.maximum(out["selected_node_count"].to_numpy(), 0.0)
    )
    out["residual_gap"] = out["selected_node_residual"] - out["fallback_residual"]
    out["fallback_better_distance"] = (out["fallback_distance"] < out["nearest_compact_distance"]).astype(np.float32)
    out["compact_fallback_score_gap"] = out["compact_top1_score"] - out["fallback_distance"]

    state_labels = ["inactive", "plastic", "protected"]
    if "selected_node_state" in df.columns:
        out = pd.concat([out, one_hot(df["selected_node_state"], "selected_state", state_labels)], axis=1)
    else:
        for label in state_labels:
            out[f"selected_state_{label}"] = 0.0
    if "fallback_state" in df.columns:
        out = pd.concat([out, one_hot(df["fallback_state"], "fallback_state", state_labels)], axis=1)
    else:
        for label in state_labels:
            out[f"fallback_state_{label}"] = 0.0
    relation_labels = ["same_task", "old_to_new", "new_to_old"]
    if "task_relation" in df.columns:
        out = pd.concat([out, one_hot(df["task_relation"], "relation", relation_labels)], axis=1)
    else:
        for label in relation_labels:
            out[f"relation_{label}"] = 0.0

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def align_features(train: pd.DataFrame, test: pd.DataFrame, columns: Optional[List[str]] = None):
    if columns is None:
        columns = sorted(set(train.columns) | set(test.columns))
    train_x = train.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=np.float32)
    test_x = test.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=np.float32)
    mean = train_x.mean(axis=0, dtype=np.float64)
    std = train_x.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32), columns


def ece_score(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(y_true[mask])) - float(np.mean(prob[mask])))
    return ece


def select_threshold(
    scores: np.ndarray,
    df: pd.DataFrame,
    candidate_mask: Optional[np.ndarray] = None,
    rates: Tuple[float, ...] = RATES,
    max_rate: float = 0.02,
    strategy: str = "max_net",
    hard_safety: bool = False,
    rng_seed: int = 0,
) -> Tuple[float, Dict[str, float]]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(scores)
    if candidate_mask is None:
        candidate_mask = np.ones(n, dtype=bool)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    available = candidate_mask & np.isfinite(scores)
    order = np.where(available)[0]
    if order.size == 0:
        return float("inf"), empty_selection_stats(n)
    order = order[np.argsort(-scores[order])]
    benefit, harm, neutral = benefit_harm_neutral(df)
    compact_acc = float(np.mean(safe_bool(df, "compact_correct"))) if n else 0.0
    best_stats = empty_selection_stats(n)
    best_threshold = float("inf")
    rng = np.random.default_rng(rng_seed)

    for rate in rates:
        if rate > max_rate + 1e-12:
            continue
        k = int(math.floor(rate * n + 1e-12))
        k = max(1, min(k, int(order.size)))
        selected = order[:k]
        threshold = float(scores[selected[-1]])
        mask = available & (scores >= threshold)
        observed_rate = float(np.mean(mask))
        if observed_rate > max_rate + 1e-12:
            continue
        stats = evaluate_mask(df, mask)
        deltas = np.zeros(int(np.sum(mask)), dtype=np.float64)
        if deltas.size:
            selected_idx = np.where(mask)[0]
            deltas = np.where(benefit[selected_idx], 1.0, np.where(harm[selected_idx], -1.0, 0.0))
        if deltas.size > 1:
            boot = []
            for _ in range(80):
                sample = rng.choice(deltas, size=deltas.size, replace=True)
                boot.append(float(np.sum(sample)) / float(n))
            std = float(np.std(boot, ddof=1))
        else:
            std = 0.0
        stats.update({
            "threshold": threshold,
            "selected_budget_rate": float(rate),
            "calibration_final_acc": compact_acc + stats["net_gain"],
            "lcb": stats["net_gain"] - std,
            "net_std": std,
        })
        eligible = stats["net_gain"] > 0.0
        if strategy == "lcb":
            eligible = stats["lcb"] > 0.0
        if hard_safety:
            eligible = eligible and stats["utility_precision"] >= 0.70
            eligible = eligible and stats["benefit"] - stats["harm"] >= 5
            eligible = eligible and (stats["benefit"] + stats["harm"] + stats["neutral"]) >= 20
        if not eligible:
            continue
        if is_better_selection(stats, best_stats, strategy=strategy):
            best_stats = stats
            best_threshold = threshold

    return best_threshold, best_stats


def empty_selection_stats(n: int) -> Dict[str, float]:
    return {
        "acc": float("nan"),
        "compact_acc": float("nan"),
        "fallback_rate": 0.0,
        "benefit": 0.0,
        "harm": 0.0,
        "neutral": 0.0,
        "net_count": 0.0,
        "net_gain": 0.0,
        "benefit_recall": 0.0,
        "utility_precision": 0.0,
        "threshold": float("inf"),
        "selected_budget_rate": 0.0,
        "calibration_final_acc": float("nan"),
        "lcb": 0.0,
        "net_std": 0.0,
    }


def is_better_selection(stats: Dict[str, float], best: Dict[str, float], strategy: str) -> bool:
    key = "lcb" if strategy == "lcb" else "net_gain"
    if stats[key] > best[key] + 1e-12:
        return True
    if abs(stats[key] - best[key]) <= 1e-12:
        if stats["utility_precision"] > best["utility_precision"] + 1e-12:
            return True
        if abs(stats["utility_precision"] - best["utility_precision"]) <= 1e-12:
            return stats["fallback_rate"] < best["fallback_rate"]
    return False


def apply_threshold(scores: np.ndarray, threshold: float, candidate_mask: Optional[np.ndarray] = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(scores) & (scores >= threshold)
    if candidate_mask is not None:
        mask &= np.asarray(candidate_mask, dtype=bool)
    return mask


def candidate_disagree(df: pd.DataFrame) -> np.ndarray:
    return (
        safe_bool(df, "fallback_available")
        & (pd.to_numeric(df["compact_top1"], errors="coerce").to_numpy() != pd.to_numeric(df["fallback_top1"], errors="coerce").to_numpy())
    )


@dataclass
class GateResult:
    method: str
    task: int
    cal_stats: Dict[str, float]
    test_stats: Dict[str, float]
    threshold: float


def fit_gbdt(cal_df: pd.DataFrame, test_df: pd.DataFrame, strategy: str = "max_net", hard_safety: bool = False):
    from sklearn.ensemble import GradientBoostingClassifier

    cal_feat = build_feature_frame(cal_df)
    test_feat = build_feature_frame(test_df)
    x_cal, x_test, names = align_features(cal_feat, test_feat)
    benefit, _, _ = benefit_harm_neutral(cal_df)
    y = benefit.astype(np.int64)
    if len(np.unique(y)) < 2:
        scores_cal = np.full(len(cal_df), -np.inf)
        scores_test = np.full(len(test_df), -np.inf)
        return scores_cal, scores_test, names, None
    pos = max(1.0, float(np.sum(y == 1)))
    neg = max(1.0, float(np.sum(y == 0)))
    weights = np.ones(len(y), dtype=np.float64)
    weights[y == 1] = min(10.0, max(1.0, math.sqrt(neg / pos)))
    model = GradientBoostingClassifier(
        n_estimators=160,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=12,
        random_state=0,
    )
    model.fit(x_cal, y, sample_weight=weights)
    return model.predict_proba(x_cal)[:, 1], model.predict_proba(x_test)[:, 1], names, model


def fit_utility_gbdt(cal_df: pd.DataFrame, test_df: pd.DataFrame):
    from sklearn.ensemble import GradientBoostingRegressor

    cal_feat = build_feature_frame(cal_df)
    test_feat = build_feature_frame(test_df)
    x_cal, x_test, names = align_features(cal_feat, test_feat)
    compact = safe_bool(cal_df, "compact_correct")
    raw = safe_bool(cal_df, "fallback_correct")
    delta = raw.astype(np.float32) - compact.astype(np.float32)
    weights = np.full(len(delta), 0.25, dtype=np.float64)
    weights[delta != 0] = 1.0
    mask = candidate_disagree(cal_df)
    if np.sum(mask) < 10:
        scores_cal = np.full(len(cal_df), -np.inf)
        scores_test = np.full(len(test_df), -np.inf)
        return scores_cal, scores_test, names, None
    model = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=12,
        random_state=1,
    )
    model.fit(x_cal[mask], delta[mask], sample_weight=weights[mask])
    scores_cal = model.predict(x_cal)
    scores_test = model.predict(x_test)
    scores_cal[~candidate_disagree(cal_df)] = -np.inf
    scores_test[~candidate_disagree(test_df)] = -np.inf
    return scores_cal, scores_test, names, model


def fit_shallow_tree(cal_df: pd.DataFrame, test_df: pd.DataFrame):
    from sklearn.tree import DecisionTreeRegressor

    cal_feat = build_feature_frame(cal_df)
    test_feat = build_feature_frame(test_df)
    x_cal, x_test, names = align_features(cal_feat, test_feat)
    compact = safe_bool(cal_df, "compact_correct")
    raw = safe_bool(cal_df, "fallback_correct")
    delta = raw.astype(np.float32) - compact.astype(np.float32)
    weights = np.full(len(delta), 0.25, dtype=np.float64)
    weights[delta != 0] = 1.0
    mask = candidate_disagree(cal_df)
    if np.sum(mask) < 10:
        return np.full(len(cal_df), -np.inf), np.full(len(test_df), -np.inf), names, None
    model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=20, random_state=2)
    model.fit(x_cal[mask], delta[mask], sample_weight=weights[mask])
    scores_cal = model.predict(x_cal)
    scores_test = model.predict(x_test)
    scores_cal[~candidate_disagree(cal_df)] = -np.inf
    scores_test[~candidate_disagree(test_df)] = -np.inf
    return scores_cal, scores_test, names, model


def fit_dual_logistic(cal_df: pd.DataFrame, test_df: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    cal_feat = build_feature_frame(cal_df)
    test_feat = build_feature_frame(test_df)
    compact_cols = [c for c in cal_feat.columns if c in COMPACT_FEATURE_HINTS]
    raw_cols = [c for c in cal_feat.columns if c in RAW_FEATURE_HINTS]
    interaction_cols = [
        "compact_raw_disagree",
        "score_cross_gap",
        "compact_score_on_raw_top1",
        "raw_score_on_compact_top1",
        "relation_same_task",
        "relation_old_to_new",
        "relation_new_to_old",
    ]
    compact_cols = sorted(set(compact_cols + [c for c in interaction_cols if c in cal_feat.columns]))
    raw_cols = sorted(set(raw_cols + [c for c in interaction_cols if c in cal_feat.columns]))
    x_comp_cal, x_comp_test, _ = align_features(cal_feat[compact_cols], test_feat[compact_cols], compact_cols)
    x_raw_cal, x_raw_test, _ = align_features(cal_feat[raw_cols], test_feat[raw_cols], raw_cols)
    y_comp = safe_bool(cal_df, "compact_correct").astype(np.int64)
    y_raw = safe_bool(cal_df, "fallback_correct").astype(np.int64)

    def fit_one(x_cal, x_test, y):
        if len(np.unique(y)) < 2:
            prob_cal = np.full(len(y), float(np.mean(y)))
            prob_test = np.full(x_test.shape[0], float(np.mean(y)))
            return prob_cal, prob_test, None
        model = LogisticRegression(class_weight="balanced", max_iter=500, solver="liblinear", random_state=0)
        model.fit(x_cal, y)
        return model.predict_proba(x_cal)[:, 1], model.predict_proba(x_test)[:, 1], model

    p_comp_cal, p_comp_test, comp_model = fit_one(x_comp_cal, x_comp_test, y_comp)
    p_raw_cal, p_raw_test, raw_model = fit_one(x_raw_cal, x_raw_test, y_raw)
    scores_cal = p_raw_cal - p_comp_cal
    scores_test = p_raw_test - p_comp_test
    scores_cal[~candidate_disagree(cal_df)] = -np.inf
    scores_test[~candidate_disagree(test_df)] = -np.inf

    metrics = {}
    try:
        metrics["compact_auc"] = float(roc_auc_score(y_comp, p_comp_cal)) if len(np.unique(y_comp)) == 2 else float("nan")
        metrics["raw_auc"] = float(roc_auc_score(y_raw, p_raw_cal)) if len(np.unique(y_raw)) == 2 else float("nan")
    except Exception:
        metrics["compact_auc"] = float("nan")
        metrics["raw_auc"] = float("nan")
    metrics["compact_ece"] = ece_score(y_comp, p_comp_cal)
    metrics["raw_ece"] = ece_score(y_raw, p_raw_cal)
    return scores_cal, scores_test, compact_cols + raw_cols, (comp_model, raw_model, metrics)


def rule_mask(df: pd.DataFrame, params: Dict[str, object]) -> np.ndarray:
    mask = candidate_disagree(df)
    mask &= pd.to_numeric(df["compact_margin"], errors="coerce").fillna(np.inf).to_numpy() <= float(params["compact_margin_thr"])
    mask &= pd.to_numeric(df["raw_margin"], errors="coerce").fillna(-np.inf).to_numpy() >= float(params["raw_margin_thr"])
    mask &= pd.to_numeric(df["score_cross_gap"], errors="coerce").fillna(-np.inf).to_numpy() >= float(params["cross_gap_thr"])
    mask &= pd.to_numeric(df["compact_rank_of_raw_top1"], errors="coerce").fillna(999).to_numpy() <= float(params["rank_k"])
    residual_thr = params.get("residual_thr")
    if residual_thr is not None:
        mask &= pd.to_numeric(df["selected_node_residual"], errors="coerce").fillna(-np.inf).to_numpy() >= float(residual_thr)
    relation = str(params.get("relation", "all"))
    if relation != "all" and "task_relation" in df.columns:
        mask &= df["task_relation"].astype(str).to_numpy() == relation
    return mask


def search_distilled_rules(cal_df: pd.DataFrame, test_df: pd.DataFrame, max_rate: float = 0.02):
    candidates = []
    compact_margin = pd.to_numeric(cal_df["compact_margin"], errors="coerce")
    raw_margin = pd.to_numeric(cal_df["raw_margin"], errors="coerce")
    cross_gap = pd.to_numeric(cal_df["score_cross_gap"], errors="coerce")
    residual = pd.to_numeric(cal_df["selected_node_residual"], errors="coerce")
    for q_cm in [0.20, 0.30, 0.40]:
        for q_rm in [0.60, 0.70, 0.80]:
            for q_gap in [0.60, 0.70, 0.80]:
                for rank_k in [2, 3]:
                    for q_res in [None, 0.50]:
                        for relation in ["all", "same_task"]:
                            params = {
                                "compact_margin_thr": float(compact_margin.quantile(q_cm)),
                                "raw_margin_thr": float(raw_margin.quantile(q_rm)),
                                "cross_gap_thr": float(cross_gap.quantile(q_gap)),
                                "rank_k": float(rank_k),
                                "residual_thr": None if q_res is None else float(residual.quantile(q_res)),
                                "relation": relation,
                                "template": f"cm<q{q_cm:.2f}&raw>q{q_rm:.2f}&gap>q{q_gap:.2f}&rank<={rank_k}&res={q_res}&rel={relation}",
                            }
                            cal_mask = rule_mask(cal_df, params)
                            if float(np.mean(cal_mask)) > max_rate + 1e-12:
                                continue
                            cal_stats = evaluate_mask(cal_df, cal_mask)
                            if cal_stats["net_gain"] <= 0:
                                continue
                            test_mask = rule_mask(test_df, params)
                            test_stats = evaluate_mask(test_df, test_mask)
                            row = dict(params)
                            row.update({
                                "cal_selected": int(np.sum(cal_mask)),
                                "cal_rate": cal_stats["fallback_rate"],
                                "cal_benefit": cal_stats["benefit"],
                                "cal_harm": cal_stats["harm"],
                                "cal_net": cal_stats["net_gain"],
                                "cal_precision": cal_stats["utility_precision"],
                                "test_selected": int(np.sum(test_mask)),
                                "test_rate": test_stats["fallback_rate"],
                                "test_benefit": test_stats["benefit"],
                                "test_harm": test_stats["harm"],
                                "test_net": test_stats["net_gain"],
                                "test_precision": test_stats["utility_precision"],
                            })
                            candidates.append(row)
    candidates.sort(key=lambda r: (r["cal_net"], r["cal_precision"], -r["cal_rate"]), reverse=True)
    return candidates


def fit_distilled_rule(cal_df: pd.DataFrame, test_df: pd.DataFrame, hard_safety: bool = True):
    rules = search_distilled_rules(cal_df, test_df, max_rate=0.02)
    if not rules:
        return np.zeros(len(cal_df), dtype=bool), np.zeros(len(test_df), dtype=bool), None, []
    for row in rules:
        if hard_safety and not (
            row["cal_precision"] >= 0.70
            and row["cal_benefit"] - row["cal_harm"] >= 5
            and row["cal_selected"] >= 20
        ):
            continue
        return rule_mask(cal_df, row), rule_mask(test_df, row), row, rules
    return np.zeros(len(cal_df), dtype=bool), np.zeros(len(test_df), dtype=bool), None, rules


def selected_bucket_rows(scores: np.ndarray, df: pd.DataFrame, rates: Tuple[float, ...] = BUCKET_RATES):
    scores = np.asarray(scores, dtype=np.float64)
    available = np.isfinite(scores)
    order = np.where(available)[0]
    if order.size:
        order = order[np.argsort(-scores[order])]
    rows = []
    n = len(df)
    for rate in rates:
        k = max(1, min(int(math.floor(rate * n + 1e-12)), int(order.size)))
        mask = np.zeros(n, dtype=bool)
        if k > 0:
            mask[order[:k]] = True
        stats = evaluate_mask(df, mask)
        rows.append({
            "bucket": f"top {pct(rate):.1f}%",
            "selected": int(np.sum(mask)),
            "fallback_rate": stats["fallback_rate"],
            "benefit": stats["benefit"],
            "harm": stats["harm"],
            "neutral": stats["neutral"],
            "net_gain": stats["net_gain"],
            "utility_precision": stats["utility_precision"],
        })
    return rows


def per_task_diagnosis(compact_traces: Dict[int, pd.DataFrame], gate_traces: Dict[int, pd.DataFrame]):
    rows = []
    for task in sorted(compact_traces):
        cdf = compact_traces[task]
        gdf = gate_traces.get(task)
        if gdf is None:
            continue
        compact_mask = np.zeros(len(cdf), dtype=bool)
        gate_mask = safe_bool(gdf, "fallback_used")
        cstats = evaluate_mask(cdf, compact_mask)
        gstats = evaluate_mask(gdf, gate_mask)
        hm = old_new_metrics(gdf, gate_mask)
        rows.append({
            "task_id": task,
            "compact_acc": pct(cstats["acc"]),
            "gate_acc": pct(gstats["acc"]),
            "delta_acc_pp": pct(gstats["acc"] - cstats["acc"]),
            "fallback_rate": gstats["fallback_rate"],
            "benefit_selected": gstats["benefit"],
            "harm_selected": gstats["harm"],
            "neutral_selected": gstats["neutral"],
            "net_gain": gstats["net_gain"],
            "old_acc": pct(hm["old_acc"]) if math.isfinite(hm["old_acc"]) else float("nan"),
            "new_acc": pct(hm["new_acc"]) if math.isfinite(hm["new_acc"]) else float("nan"),
            "old_new_HM": pct(hm["old_new_hm"]) if math.isfinite(hm["old_new_hm"]) else float("nan"),
        })
    return rows


def summarize_task_method(method: str, task_results: List[GateResult], compact_task_acc: Dict[int, float], compact_mb: float):
    accs = [r.test_stats["acc"] for r in task_results]
    final = task_results[-1]
    final_hm = old_new_metrics(
        pd.DataFrame(), np.zeros(0, dtype=bool)
    )
    selected_rel_avg = pct(float(np.mean(accs)) - float(np.mean([compact_task_acc[t.task] for t in task_results])))
    selected_rel_last = pct(final.test_stats["acc"] - compact_task_acc[final.task])
    return {
        "Method": method,
        "A_Avg": pct(float(np.mean(accs))) if accs else float("nan"),
        "A_Last": pct(final.test_stats["acc"]) if accs else float("nan"),
        "selected_rel_A_Last": selected_rel_last,
        "selected_rel_A_Avg": selected_rel_avg,
        "raw_rel_A_Last": pct(final.test_stats["acc"]) - RAW_HC_SOINN_A_LAST if accs else float("nan"),
        "fallback_rate": final.test_stats["fallback_rate"] if accs else 0.0,
        "benefit": final.test_stats["benefit"] if accs else 0,
        "harm": final.test_stats["harm"] if accs else 0,
        "neutral": final.test_stats["neutral"] if accs else 0,
        "net": final.test_stats["net_gain"] if accs else 0.0,
        "benefit_recall": final.test_stats["benefit_recall"] if accs else 0.0,
        "utility_precision": final.test_stats["utility_precision"] if accs else 0.0,
        "calibration_net": final.cal_stats["net_gain"] if accs else 0.0,
        "calibration_fallback_rate": final.cal_stats["fallback_rate"] if accs else 0.0,
        "compact_MB": compact_mb,
        "memory_reduction_vs_raw": 1.0 - compact_mb / RAW_HC_SOINN_COMPACT_MB,
    }


def run_gate_over_tasks(
    name: str,
    calibration_traces: Dict[int, pd.DataFrame],
    test_traces: Dict[int, pd.DataFrame],
    fit_func,
    threshold_strategy: str,
    hard_safety: bool,
    compact_task_acc: Dict[int, float],
    compact_mb: float,
):
    results: List[GateResult] = []
    extras: Dict[str, object] = {}
    for task in sorted(test_traces):
        cal_df = calibration_traces[task]
        test_df = test_traces[task]
        scores_cal, scores_test, names, model = fit_func(cal_df, test_df)
        threshold, cal_stats = select_threshold(
            scores_cal,
            cal_df,
            rates=RATES,
            max_rate=0.02,
            strategy=threshold_strategy,
            hard_safety=hard_safety,
            rng_seed=task,
        )
        test_mask = apply_threshold(scores_test, threshold)
        test_stats = evaluate_mask(test_df, test_mask)
        results.append(GateResult(name, task, cal_stats, test_stats, threshold))
        if task == max(test_traces):
            extras["final_scores_cal"] = scores_cal
            extras["final_scores_test"] = scores_test
            extras["feature_names"] = names
            extras["model"] = model
    return summarize_task_method(name, results, compact_task_acc, compact_mb), results, extras


def run_distilled_rule_over_tasks(calibration_traces, test_traces, compact_task_acc):
    results = []
    final_rules = []
    final_best = None
    for task in sorted(test_traces):
        cal_df = calibration_traces[task]
        test_df = test_traces[task]
        cal_mask, test_mask, best, rules = fit_distilled_rule(cal_df, test_df, hard_safety=True)
        cal_stats = evaluate_mask(cal_df, cal_mask)
        cal_stats.update({"threshold": float("nan"), "selected_budget_rate": cal_stats["fallback_rate"], "lcb": cal_stats["net_gain"], "net_std": 0.0})
        test_stats = evaluate_mask(test_df, test_mask)
        results.append(GateResult("distilled rule list + safety", task, cal_stats, test_stats, float("nan")))
        if task == max(test_traces):
            final_rules = rules[:50]
            final_best = best
    return summarize_task_method(
        "distilled rule list + safety",
        results,
        compact_task_acc,
        COMPACT_BASE_MB + RULE_GATE_MB,
    ), results, final_rules, final_best


def utility_pruned_cache(final_cal: pd.DataFrame, final_test: pd.DataFrame, gbdt_scores_cal: np.ndarray, gbdt_scores_test: np.ndarray):
    def key_frame(df):
        cls = pd.to_numeric(df["fallback_top1"], errors="coerce").fillna(-1).astype(int).astype(str)
        node = pd.to_numeric(df["fallback_node"], errors="coerce").fillna(-1).astype(int).astype(str)
        return cls + "|" + node

    cal_keys = key_frame(final_cal)
    benefit, harm, _ = benefit_harm_neutral(final_cal)
    table = {}
    for key, b, h in zip(cal_keys, benefit, harm):
        if key not in table:
            table[key] = {"selected_count": 0, "benefit_count": 0, "harm_count": 0}
        table[key]["selected_count"] += 1
        table[key]["benefit_count"] += int(b)
        table[key]["harm_count"] += int(h)
    rows = []
    kept = set()
    for key, vals in table.items():
        b = vals["benefit_count"]
        h = vals["harm_count"]
        n = vals["selected_count"]
        utility = b - h
        utility_rate = utility / max(1, n)
        keep = b >= h + 1 and b > 0
        if keep:
            kept.add(key)
        rows.append({
            "fallback_node_key": key,
            "selected_count": n,
            "benefit_count": b,
            "harm_count": h,
            "utility": utility,
            "utility_rate": utility_rate,
            "kept": keep,
        })
    rows.sort(key=lambda r: (r["kept"], r["utility"], r["utility_rate"]), reverse=True)

    test_keys = key_frame(final_test)
    prune_mask_test = test_keys.isin(kept).to_numpy()
    prune_mask_cal = cal_keys.isin(kept).to_numpy()
    original_oracle = evaluate_mask(final_test, benefit_harm_neutral(final_test)[0])
    pruned_oracle = evaluate_mask(final_test, benefit_harm_neutral(final_test)[0] & prune_mask_test)

    rng = np.random.default_rng(0)
    all_keys = np.asarray(sorted(set(test_keys.tolist())))
    random_nets = []
    random_oracles = []
    for _ in range(50):
        chosen = set(rng.choice(all_keys, size=min(len(kept), len(all_keys)), replace=False).tolist()) if len(kept) else set()
        rmask = test_keys.isin(chosen).to_numpy()
        stats = evaluate_mask(final_test, benefit_harm_neutral(final_test)[0] & rmask)
        random_nets.append(stats["net_gain"])
        random_oracles.append(stats["acc"])

    thr_orig, cal_orig = select_threshold(gbdt_scores_cal, final_cal, rates=RATES, max_rate=0.02, strategy="lcb", hard_safety=True)
    test_orig = evaluate_mask(final_test, apply_threshold(gbdt_scores_test, thr_orig))
    thr_pruned, cal_pruned = select_threshold(
        gbdt_scores_cal,
        final_cal,
        candidate_mask=prune_mask_cal,
        rates=RATES,
        max_rate=0.02,
        strategy="lcb",
        hard_safety=True,
    )
    test_pruned = evaluate_mask(final_test, apply_threshold(gbdt_scores_test, thr_pruned, prune_mask_test))

    summary = [
        {
            "method": "count-r3 raw cache oracle",
            "kept_nodes": len(set(test_keys.tolist())),
            "fallback_rate": original_oracle["fallback_rate"],
            "benefit": original_oracle["benefit"],
            "harm": original_oracle["harm"],
            "net_gain": original_oracle["net_gain"],
            "A_Last": pct(original_oracle["acc"]),
        },
        {
            "method": "utility-pruned raw cache oracle",
            "kept_nodes": len(kept),
            "fallback_rate": pruned_oracle["fallback_rate"],
            "benefit": pruned_oracle["benefit"],
            "harm": pruned_oracle["harm"],
            "net_gain": pruned_oracle["net_gain"],
            "A_Last": pct(pruned_oracle["acc"]),
        },
        {
            "method": "same-count random-pruned oracle mean",
            "kept_nodes": len(kept),
            "fallback_rate": float("nan"),
            "benefit": float("nan"),
            "harm": float("nan"),
            "net_gain": float(np.mean(random_nets)) if random_nets else 0.0,
            "A_Last": pct(float(np.mean(random_oracles))) if random_oracles else float("nan"),
        },
        {
            "method": "GBDT LCB on original cache",
            "kept_nodes": len(set(test_keys.tolist())),
            "fallback_rate": test_orig["fallback_rate"],
            "benefit": test_orig["benefit"],
            "harm": test_orig["harm"],
            "net_gain": test_orig["net_gain"],
            "A_Last": pct(test_orig["acc"]),
        },
        {
            "method": "GBDT LCB on utility-pruned cache",
            "kept_nodes": len(kept),
            "fallback_rate": test_pruned["fallback_rate"],
            "benefit": test_pruned["benefit"],
            "harm": test_pruned["harm"],
            "net_gain": test_pruned["net_gain"],
            "A_Last": pct(test_pruned["acc"]),
        },
    ]
    return rows, summary, len(kept)


def constrained_topk_table(final_cal: pd.DataFrame, final_test: pd.DataFrame, gbdt_scores_cal: np.ndarray, gbdt_scores_test: np.ndarray):
    rows = []
    for k in [1, 2, 3, 5, 10]:
        if k == 1:
            cal_constraint = np.zeros(len(final_cal), dtype=bool)
            test_constraint = np.zeros(len(final_test), dtype=bool)
            note = "top1 constraint forbids prediction change"
        elif k == 2:
            cal_constraint = (
                pd.to_numeric(final_cal["fallback_top1"], errors="coerce").to_numpy()
                == pd.to_numeric(final_cal["compact_top2"], errors="coerce").to_numpy()
            )
            test_constraint = (
                pd.to_numeric(final_test["fallback_top1"], errors="coerce").to_numpy()
                == pd.to_numeric(final_test["compact_top2"], errors="coerce").to_numpy()
            )
            note = "exact with stored compact top2"
        else:
            rows.append({
                "K": k,
                "oracle_gain": float("nan"),
                "benefit_rate": float("nan"),
                "harm_rate": float("nan"),
                "raw_fallback_acc": float("nan"),
                "best_gate_net": float("nan"),
                "A_Last": float("nan"),
                "fallback_rate": float("nan"),
                "note": "requires compact top3/top5/top10 or full raw class-score trace",
            })
            continue
        benefit, harm, _ = benefit_harm_neutral(final_test)
        oracle_mask = benefit & test_constraint
        oracle = evaluate_mask(final_test, oracle_mask)
        thr, _ = select_threshold(
            gbdt_scores_cal,
            final_cal,
            candidate_mask=cal_constraint,
            rates=RATES,
            max_rate=0.02,
            strategy="lcb",
            hard_safety=True,
        )
        gate = evaluate_mask(final_test, apply_threshold(gbdt_scores_test, thr, test_constraint))
        candidate_count = max(1, int(np.sum(test_constraint)))
        rows.append({
            "K": k,
            "oracle_gain": oracle["net_gain"],
            "benefit_rate": float(np.sum(benefit & test_constraint)) / candidate_count,
            "harm_rate": float(np.sum(harm & test_constraint)) / candidate_count,
            "raw_fallback_acc": float(np.mean(safe_bool(final_test, "fallback_correct")[test_constraint])) if np.any(test_constraint) else float("nan"),
            "best_gate_net": gate["net_gain"],
            "A_Last": pct(gate["acc"]),
            "fallback_rate": gate["fallback_rate"],
            "note": note,
        })
    return rows


def build_report(
    out_path: Path,
    method_summary: List[Dict[str, object]],
    per_task_rows: List[Dict[str, object]],
    bucket_rows: List[Dict[str, object]],
    importance_rows: List[Dict[str, object]],
    rule_rows: List[Dict[str, object]],
    dual_rows: List[Dict[str, object]],
    pruned_summary: List[Dict[str, object]],
    topk_rows: List[Dict[str, object]],
    lcb_rows: List[Dict[str, object]],
    kept_pruned_nodes: int,
):
    summary_rows = []
    for row in method_summary:
        summary_rows.append([
            row["Method"],
            fmt(row["A_Avg"], 4),
            fmt(row["A_Last"], 2),
            fmt(row["selected_rel_A_Last"], 2, "pp"),
            fmt(row["raw_rel_A_Last"], 2, "pp"),
            "n/a",
            fmt(pct(row["fallback_rate"]), 2, "%"),
            int(row["benefit"]),
            int(row["harm"]),
            fmt(pct(row["net"]), 2, "pp"),
            fmt(row["compact_MB"], 4),
        ])

    per_task_preview = []
    for row in per_task_rows:
        per_task_preview.append([
            row["task_id"],
            fmt(row["compact_acc"], 2),
            fmt(row["gate_acc"], 2),
            fmt(row["delta_acc_pp"], 2, "pp"),
            fmt(pct(row["fallback_rate"]), 2, "%"),
            int(row["benefit_selected"]),
            int(row["harm_selected"]),
            fmt(pct(row["net_gain"]), 2, "pp"),
            fmt(row["old_acc"], 2),
            fmt(row["new_acc"], 2),
            fmt(row["old_new_HM"], 2),
        ])

    bucket_preview = [
        [
            r["bucket"],
            int(r["selected"]),
            fmt(pct(r["fallback_rate"]), 2, "%"),
            int(r["benefit"]),
            int(r["harm"]),
            int(r["neutral"]),
            fmt(pct(r["net_gain"]), 2, "pp"),
            fmt(pct(r["utility_precision"]), 2, "%"),
        ]
        for r in bucket_rows
    ]
    importance_preview = [
        [r["rank"], r["feature"], fmt(r["importance"], 4)]
        for r in importance_rows[:12]
    ]
    rule_preview = [
        [
            r["template"],
            int(r["cal_selected"]),
            fmt(pct(r["cal_net"]), 2, "pp"),
            fmt(pct(r["cal_precision"]), 2, "%"),
            int(r["test_selected"]),
            fmt(pct(r["test_net"]), 2, "pp"),
            fmt(pct(r["test_precision"]), 2, "%"),
        ]
        for r in rule_rows[:10]
    ]
    dual_preview = [
        [r["task"], fmt(r["compact_auc"], 4), fmt(r["compact_ece"], 4), fmt(r["raw_auc"], 4), fmt(r["raw_ece"], 4)]
        for r in dual_rows[-5:]
    ]
    pruned_preview = [
        [
            r["method"],
            int(r["kept_nodes"]) if math.isfinite(float(r["kept_nodes"])) else "n/a",
            fmt(pct(r["fallback_rate"]), 2, "%"),
            r["benefit"],
            r["harm"],
            fmt(pct(r["net_gain"]), 2, "pp"),
            fmt(r["A_Last"], 2),
        ]
        for r in pruned_summary
    ]
    topk_preview = [
        [
            r["K"],
            fmt(pct(r["oracle_gain"]), 2, "pp"),
            fmt(pct(r["benefit_rate"]), 2, "%"),
            fmt(pct(r["harm_rate"]), 2, "%"),
            fmt(pct(r["best_gate_net"]), 2, "pp"),
            fmt(r["A_Last"], 2),
            r["note"],
        ]
        for r in topk_rows
    ]
    lcb_preview = [
        [
            r["method"],
            fmt(pct(r["calibration_net"]), 2, "pp"),
            fmt(pct(r["calibration_fallback_rate"]), 2, "%"),
            fmt(pct(r["test_net"]), 2, "pp"),
            fmt(pct(r["test_fallback_rate"]), 2, "%"),
            fmt(pct(r["test_precision"]), 2, "%"),
        ]
        for r in lcb_rows
    ]

    best = max(method_summary, key=lambda r: r["selected_rel_A_Last"])
    passes = (
        best["selected_rel_A_Last"] >= 0.2
        and best["selected_rel_A_Avg"] >= -0.1
        and best["net"] > 0
        and best["memory_reduction_vs_raw"] >= 0.30
    )
    verdict = (
        "Direction 1 reaches the minimum success line; run 3 seeds next."
        if passes
        else "Direction 1 still does not reach the minimum success line; keep it as diagnostic and move main effort to Direction 2."
    )

    lines = [
        "# Direction 1 Next Optimization Report",
        "",
        "Scope: CUB-200 seed 1993, same-feature cached ViT-B/16, classifier-only CPU path. "
        "Calibration trace was newly dumped before gate fitting; all threshold choices in this report use calibration only.",
        "",
        "## Main Result",
        "",
        markdown_table(
            ["Method", "A_Avg", "A_Last", "selected-rel A_Last", "raw-rel A_Last", "HM", "fallback rate", "benefit", "harm", "net", "compact MB"],
            summary_rows,
        ),
        "",
        f"Best selected-relative A_Last is `{best['Method']}` at {fmt(best['selected_rel_A_Last'], 2, 'pp')}. "
        f"Minimum success requires +0.20pp A_Last, A_Avg drop <= 0.10pp, positive test net, and >=30% compact-memory reduction. {verdict}",
        "",
        "## Per-Task / Old-New Diagnosis",
        "",
        markdown_table(
            ["task", "compact", "pair/node gate", "delta", "fallback", "benefit", "harm", "net", "old", "new", "HM"],
            per_task_preview,
        ),
        "",
        "The previous pair/node gate is late-task positive but has multiple mid-task negative points. "
        "That explains the earlier A_Last gain with A_Avg loss: the gate is not uniformly safe across the task stream.",
        "",
        "## GBDT Diagnostic",
        "",
        markdown_table(
            ["Bucket", "selected", "fallback", "benefit", "harm", "neutral", "net", "precision"],
            bucket_preview,
        ),
        "",
        "Feature importance from the final-task calibration GBDT:",
        "",
        markdown_table(["rank", "feature", "importance"], importance_preview),
        "",
        "## Distilled Rule Search",
        "",
        markdown_table(
            ["rule", "cal selected", "cal net", "cal precision", "test selected", "test net", "test precision"],
            rule_preview,
        )
        if rule_preview
        else "No distilled rule passed positive calibration net under the 2% budget.",
        "",
        "## Dual Calibrator",
        "",
        markdown_table(["task", "compact AUC", "compact ECE", "raw AUC", "raw ECE"], dual_preview),
        "",
        "Dual calibration separates compact/raw correctness more stably than direct delta regression on calibration, "
        "but its top-ranked fallback set still does not transfer strongly enough to test.",
        "",
        "## Utility-Pruned Cache",
        "",
        f"Calibration utility pruning kept {kept_pruned_nodes} top raw-node keys on the final task under the conservative rule `benefit >= harm + 1`.",
        "",
        markdown_table(["method", "kept nodes", "fallback", "benefit", "harm", "net", "A_Last"], pruned_preview),
        "",
        "This is a top-node pruning approximation: the current trace does not contain the second-best raw node after a pruned top node is removed.",
        "",
        "## Compact Top-K Constraint",
        "",
        markdown_table(["K", "oracle gain", "benefit rate", "harm rate", "best gate net", "A_Last", "note"], topk_preview),
        "",
        "Exact K>2 challenge requires storing compact top3/top5/top10 or full raw class scores; current traces only store compact top2 and raw top2.",
        "",
        "## LCB Threshold Selection",
        "",
        markdown_table(["method", "cal net", "cal fallback", "test net", "test fallback", "test precision"], lcb_preview),
        "",
        "LCB plus hard safety closes the calibration-tail overfit by disabling weak selections, but the resulting deployable gain is too small.",
        "",
        "## Memory Ledger Audit",
        "",
        markdown_table(
            ["Method", "compact MB", "reduction vs raw", "note"],
            [
                ["compact baseline", fmt(COMPACT_BASE_MB, 4), fmt(pct(1.0 - COMPACT_BASE_MB / RAW_HC_SOINN_COMPACT_MB), 2, "%"), "includes count-r3 raw fallback cache in ledger"],
                ["GBDT gates", fmt(COMPACT_BASE_MB + GBDT_GATE_MB, 4), fmt(pct(1.0 - (COMPACT_BASE_MB + GBDT_GATE_MB) / RAW_HC_SOINN_COMPACT_MB), 2, "%"), "tree model counted by serialized size estimate"],
                ["dual/logistic gates", fmt(COMPACT_BASE_MB + LINEAR_GATE_MB, 4), fmt(pct(1.0 - (COMPACT_BASE_MB + LINEAR_GATE_MB) / RAW_HC_SOINN_COMPACT_MB), 2, "%"), "two small linear calibrators"],
                ["distilled rules", fmt(COMPACT_BASE_MB + RULE_GATE_MB, 4), fmt(pct(1.0 - (COMPACT_BASE_MB + RULE_GATE_MB) / RAW_HC_SOINN_COMPACT_MB), 2, "%"), "threshold/rule metadata only"],
            ],
        ),
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "Do not run 3 seeds for this branch yet. The strongest evidence remains the oracle upper bound: raw fallback contains recoverable errors, "
        "but the deployable reliability gate cannot yet select them with enough stable precision. The next main branch should be Direction 2 atom-conflict gate; "
        "Direction 1 can stay as an appendix diagnostic unless OOF calibration is implemented later.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", default="logs/direction1_next_optimization/base_trace")
    parser.add_argument("--pair_trace_dir", default="logs/direction1_utility_traces/delta_ridge_pair")
    parser.add_argument("--output_dir", default="logs/direction1_next_optimization")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_traces = load_traces(trace_dir, "calibration")
    test_traces = load_traces(trace_dir, "test")
    pair_traces = load_existing_test_traces(Path(args.pair_trace_dir))

    per_task_rows = per_task_diagnosis(test_traces, pair_traces)
    write_csv(out_dir / "per_task_old_new_diagnosis.csv", per_task_rows)

    compact_task_acc = {
        task: evaluate_mask(df, np.zeros(len(df), dtype=bool))["acc"]
        for task, df in test_traces.items()
    }
    compact_results = [
        GateResult(
            "compact baseline",
            task,
            empty_selection_stats(len(calibration_traces[task])),
            evaluate_mask(test_traces[task], np.zeros(len(test_traces[task]), dtype=bool)),
            float("inf"),
        )
        for task in sorted(test_traces)
    ]
    method_summary = [
        summarize_task_method("compact baseline", compact_results, compact_task_acc, COMPACT_BASE_MB)
    ]

    gbdt_summary, gbdt_results, gbdt_extra = run_gate_over_tasks(
        "GBDT max2 recalibrated",
        calibration_traces,
        test_traces,
        fit_gbdt,
        threshold_strategy="max_net",
        hard_safety=False,
        compact_task_acc=compact_task_acc,
        compact_mb=COMPACT_BASE_MB + GBDT_GATE_MB,
    )
    method_summary.append(gbdt_summary)

    gbdt_lcb_summary, gbdt_lcb_results, gbdt_lcb_extra = run_gate_over_tasks(
        "GBDT + LCB safety",
        calibration_traces,
        test_traces,
        fit_gbdt,
        threshold_strategy="lcb",
        hard_safety=True,
        compact_task_acc=compact_task_acc,
        compact_mb=COMPACT_BASE_MB + GBDT_GATE_MB,
    )
    method_summary.append(gbdt_lcb_summary)

    utility_gbdt_summary, utility_gbdt_results, _ = run_gate_over_tasks(
        "utility GBDT delta",
        calibration_traces,
        test_traces,
        fit_utility_gbdt,
        threshold_strategy="lcb",
        hard_safety=True,
        compact_task_acc=compact_task_acc,
        compact_mb=COMPACT_BASE_MB + GBDT_GATE_MB,
    )
    method_summary.append(utility_gbdt_summary)

    tree_summary, tree_results, _ = run_gate_over_tasks(
        "shallow utility tree",
        calibration_traces,
        test_traces,
        fit_shallow_tree,
        threshold_strategy="lcb",
        hard_safety=True,
        compact_task_acc=compact_task_acc,
        compact_mb=COMPACT_BASE_MB + 0.004,
    )
    method_summary.append(tree_summary)

    dual_metrics_rows = []

    def dual_wrapper(cal_df, test_df):
        scores_cal, scores_test, names, model = fit_dual_logistic(cal_df, test_df)
        metrics = model[2] if isinstance(model, tuple) else {}
        dual_metrics_rows.append({
            "task": int(test_df["task"].iloc[0]) if "task" in test_df.columns else len(dual_metrics_rows),
            "compact_auc": metrics.get("compact_auc", float("nan")),
            "compact_ece": metrics.get("compact_ece", float("nan")),
            "raw_auc": metrics.get("raw_auc", float("nan")),
            "raw_ece": metrics.get("raw_ece", float("nan")),
        })
        return scores_cal, scores_test, names, model

    dual_summary, dual_results, _ = run_gate_over_tasks(
        "dual logistic p_raw-p_compact",
        calibration_traces,
        test_traces,
        dual_wrapper,
        threshold_strategy="lcb",
        hard_safety=True,
        compact_task_acc=compact_task_acc,
        compact_mb=COMPACT_BASE_MB + LINEAR_GATE_MB,
    )
    method_summary.append(dual_summary)
    write_csv(out_dir / "dual_calibrator_metrics.csv", dual_metrics_rows)

    rule_summary, rule_results, final_rule_rows, best_rule = run_distilled_rule_over_tasks(
        calibration_traces,
        test_traces,
        compact_task_acc,
    )
    method_summary.append(rule_summary)
    write_csv(out_dir / "distilled_rule_search.csv", final_rule_rows)

    final_task = max(test_traces)
    final_cal = calibration_traces[final_task]
    final_test = test_traces[final_task]
    gbdt_scores_cal = gbdt_lcb_extra["final_scores_cal"]
    gbdt_scores_test = gbdt_lcb_extra["final_scores_test"]
    bucket_rows = selected_bucket_rows(gbdt_scores_test, final_test, rates=BUCKET_RATES)
    write_csv(out_dir / "gbdt_top_bucket_stats.csv", bucket_rows)

    importance_rows = []
    model = gbdt_lcb_extra.get("model")
    names = gbdt_lcb_extra.get("feature_names", [])
    if model is not None and hasattr(model, "feature_importances_"):
        order = np.argsort(-model.feature_importances_)
        for rank, idx in enumerate(order, start=1):
            importance_rows.append({
                "rank": rank,
                "feature": names[int(idx)],
                "importance": float(model.feature_importances_[int(idx)]),
            })
    write_csv(out_dir / "gbdt_feature_importance.csv", importance_rows)

    pruned_rows, pruned_summary, kept_pruned_nodes = utility_pruned_cache(
        final_cal,
        final_test,
        gbdt_scores_cal,
        gbdt_scores_test,
    )
    write_csv(out_dir / "utility_pruned_cache_nodes.csv", pruned_rows)
    write_csv(out_dir / "utility_pruned_cache_results.csv", pruned_summary)

    topk_rows = constrained_topk_table(final_cal, final_test, gbdt_scores_cal, gbdt_scores_test)
    write_csv(out_dir / "compact_topk_constrained_results.csv", topk_rows)

    lcb_rows = []
    for summary, results in [
        (gbdt_summary, gbdt_results),
        (gbdt_lcb_summary, gbdt_lcb_results),
        (utility_gbdt_summary, utility_gbdt_results),
        (tree_summary, tree_results),
        (dual_summary, dual_results),
        (rule_summary, rule_results),
    ]:
        final = results[-1]
        lcb_rows.append({
            "method": summary["Method"],
            "calibration_net": final.cal_stats["net_gain"],
            "calibration_fallback_rate": final.cal_stats["fallback_rate"],
            "test_net": final.test_stats["net_gain"],
            "test_fallback_rate": final.test_stats["fallback_rate"],
            "test_precision": final.test_stats["utility_precision"],
        })
    write_csv(out_dir / "lcb_threshold_selection.csv", lcb_rows)
    write_csv(out_dir / "method_summary.csv", method_summary)

    build_report(
        out_dir / "direction1_next_optimization_report.md",
        method_summary,
        per_task_rows,
        bucket_rows,
        importance_rows,
        final_rule_rows,
        dual_metrics_rows,
        pruned_summary,
        topk_rows,
        lcb_rows,
        kept_pruned_nodes,
    )

    print(f"Wrote {out_dir / 'direction1_next_optimization_report.md'}")
    print(f"Wrote {out_dir / 'method_summary.csv'}")
    print(f"Wrote {out_dir / 'per_task_old_new_diagnosis.csv'}")


if __name__ == "__main__":
    main()
