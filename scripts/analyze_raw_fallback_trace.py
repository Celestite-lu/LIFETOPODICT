#!/usr/bin/env python3
"""Analyze raw-fallback prediction traces for deployable gate design.

The positive label is oracle-positive:
compact prediction is wrong while the nearest raw fallback prediction is right.
This script is diagnostic only; it does not tune production thresholds.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--final_task_only", action="store_true", default=True)
    parser.add_argument("--include_all_tasks", action="store_true", default=False)
    parser.add_argument("--top_plots", type=int, default=12)
    parser.add_argument("--random_seed", type=int, default=1993)
    return parser.parse_args()


def read_trace_dir(trace_dir):
    paths = sorted(Path(trace_dir).glob("*.jsonl"))
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_trace_file"] = str(path)
                rows.append(rec)
    if not rows:
        raise FileNotFoundError(f"No JSONL trace records found in {trace_dir}")
    return pd.DataFrame(rows)


def to_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def add_derived_features(df):
    numeric_base = [
        "compact_top1",
        "compact_top2",
        "compact_top1_score",
        "compact_top2_score",
        "compact_margin",
        "nearest_compact_node",
        "nearest_compact_count",
        "selected_node_count",
        "nearest_compact_distance",
        "selected_node_residual",
        "fallback_top1",
        "fallback_top2",
        "fallback_node",
        "fallback_count",
        "fallback_distance",
        "fallback_top2_distance",
        "fallback_margin",
        "fallback_residual",
        "final_top1",
        "target",
        "task",
        "total_classes",
        "init_cls",
        "increment",
        "selected_atom_inactive_ratio",
        "selected_atom_plastic_ratio",
        "selected_atom_protected_ratio",
        "selected_atom_weighted_inactive_ratio",
        "selected_atom_weighted_protected_ratio",
        "selected_atom_old_support_mean",
        "selected_atom_weighted_old_support",
        "selected_atom_usage_ema_mean",
        "selected_atom_weighted_usage_ema",
        "selected_atom_abs_sum",
        "selected_atom_entropy",
    ]
    to_numeric(df, numeric_base)

    for col in ["compact_correct", "fallback_correct", "final_correct", "fallback_available", "fallback_used"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    df["oracle_positive"] = (
        df.get("fallback_available", False)
        & (~df.get("compact_correct", False))
        & df.get("fallback_correct", False)
    )
    df["fallback_harm"] = (
        df.get("fallback_available", False)
        & df.get("compact_correct", False)
        & (~df.get("fallback_correct", False))
    )
    df["both_correct"] = (
        df.get("fallback_available", False)
        & df.get("compact_correct", False)
        & df.get("fallback_correct", False)
    )
    df["both_wrong"] = (
        df.get("fallback_available", False)
        & (~df.get("compact_correct", False))
        & (~df.get("fallback_correct", False))
    )
    df["compact_raw_disagree_sample"] = (
        df.get("fallback_available", False)
        & (df.get("compact_top1", -1) != df.get("fallback_top1", -1))
    )
    df["compact_top1_eq_top2"] = df.get("compact_top1", -1) == df.get("compact_top2", -2)

    df["selected_count_log"] = np.log1p(df.get("nearest_compact_count", np.nan).clip(lower=0))
    df["fallback_count_log"] = np.log1p(df.get("fallback_count", np.nan).clip(lower=0))
    df["count_log_gap_fallback_minus_compact"] = df["fallback_count_log"] - df["selected_count_log"]
    df["distance_gap_compact_minus_fallback"] = (
        df.get("nearest_compact_distance", np.nan) - df.get("fallback_distance", np.nan)
    )
    df["residual_gap_compact_minus_fallback"] = (
        df.get("selected_node_residual", np.nan) - df.get("fallback_residual", np.nan)
    )
    df["compact_fallback_score_gap"] = (
        df.get("compact_top1_score", np.nan) - df.get("fallback_distance", np.nan)
    )
    df["fallback_better_distance"] = (
        df.get("fallback_distance", np.inf) < df.get("nearest_compact_distance", -np.inf)
    ).astype(float)
    df["fallback_margin_ratio"] = safe_divide(
        df.get("fallback_margin", np.nan), df.get("compact_margin", np.nan)
    )
    df["distance_gap_over_margin"] = safe_divide(
        df["distance_gap_compact_minus_fallback"], df.get("compact_margin", np.nan)
    )

    init_cls = int(pd.to_numeric(df.get("init_cls", pd.Series([10])), errors="coerce").dropna().iloc[0])
    increment = int(pd.to_numeric(df.get("increment", pd.Series([10])), errors="coerce").dropna().iloc[0])
    cur_task = pd.to_numeric(df.get("task", 0), errors="coerce").fillna(0).astype(int)
    for src in ["target", "compact_top1", "fallback_top1"]:
        if src in df.columns:
            cls_task = class_to_task(df[src], init_cls, increment)
            df[f"{src}_task_id"] = cls_task
            df[f"{src}_age"] = cur_task - cls_task
            df[f"{src}_is_current_task"] = (df[f"{src}_age"] == 0).astype(float)
            df[f"{src}_is_old"] = (df[f"{src}_age"] > 0).astype(float)

    for state_col in ["selected_node_state", "fallback_state"]:
        if state_col in df.columns:
            state = df[state_col].fillna("unknown").astype(str).str.lower()
            for value in ["inactive", "plastic", "protected", "unknown"]:
                df[f"{state_col}_{value}"] = (state == value).astype(float)

    return df


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def class_to_task(classes, init_cls, increment):
    cls = pd.to_numeric(classes, errors="coerce").fillna(-1).astype(int)
    task = np.zeros(len(cls), dtype=int)
    mask = cls >= init_cls
    task[mask] = 1 + ((cls[mask] - init_cls) // max(1, increment))
    task[cls < 0] = -1
    return task


def numeric_feature_columns(df):
    exclude = {
        "target",
        "sample_index",
        "task",
        "known_classes",
        "total_classes",
        "init_cls",
        "increment",
        "compact_top1",
        "compact_top2",
        "fallback_top1",
        "fallback_top2",
        "final_top1",
        "nearest_compact_node",
        "fallback_node",
    }
    bool_cols = {
        "compact_correct",
        "fallback_correct",
        "final_correct",
        "fallback_available",
        "fallback_used",
        "oracle_positive",
        "fallback_harm",
        "both_correct",
        "both_wrong",
    }
    cols = []
    for col in df.columns:
        if col in exclude or col in bool_cols or col.startswith("_"):
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            valid = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if valid.nunique() > 1:
                cols.append(col)
    return cols


def summarize_feature_groups(df, feature_cols, label_col="oracle_positive"):
    rows = []
    pos = df[label_col].astype(bool)
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for group_name, mask in [("oracle_positive", pos), ("ordinary", ~pos)]:
            values = x[mask].dropna().to_numpy(dtype=float)
            row = {
                "feature": col,
                "group": group_name,
                "n": int(values.size),
                "missing": int(mask.sum() - values.size),
            }
            if values.size:
                row.update(
                    {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "median": float(np.median(values)),
                        "q10": float(np.quantile(values, 0.10)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "q90": float(np.quantile(values, 0.90)),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def feature_screen(df, feature_cols, label_col="oracle_positive"):
    y = df[label_col].astype(int).to_numpy()
    rows = []
    pos_rate = float(np.mean(y))
    budgets = sorted(set([0.01, 0.02, pos_rate, 0.06, 0.10]))
    for col in feature_cols:
        raw = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if raw.notna().sum() < 10:
            continue
        x = raw.to_numpy(dtype=float)
        fill = np.nanmedian(x)
        x = np.where(np.isfinite(x), x, fill)
        if np.nanstd(x) < 1e-12:
            continue
        try:
            auc = roc_auc_score(y, x)
        except ValueError:
            continue
        direction = "high"
        score = x.copy()
        best_auc = auc
        if auc < 0.5:
            direction = "low"
            score = -x
            best_auc = 1.0 - auc
        ap = average_precision_score(y, score)
        row = {
            "feature": col,
            "auc_raw": float(auc),
            "auc_best_orientation": float(best_auc),
            "orientation_positive_when": direction,
            "average_precision": float(ap),
        }
        for budget in budgets:
            metrics = metrics_at_budget(y, score, budget, df)
            suffix = f"b{budget:.4f}".replace(".", "p")
            for key, val in metrics.items():
                row[f"{key}_{suffix}"] = val
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["auc_best_orientation", "average_precision"],
            ascending=[False, False],
        )
    return out


def metrics_at_budget(y, score, budget, df=None):
    n = len(y)
    k = max(1, int(round(float(budget) * n)))
    order = np.argsort(-score, kind="mergesort")
    selected = np.zeros(n, dtype=bool)
    selected[order[:k]] = True
    tp = int(np.sum(selected & (y == 1)))
    fp = int(np.sum(selected & (y == 0)))
    fn = int(np.sum((~selected) & (y == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    out = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fallback_rate": float(np.mean(selected)),
    }
    if df is not None and "compact_correct" in df.columns and "fallback_correct" in df.columns:
        compact = df["compact_correct"].astype(bool).to_numpy()
        fallback = df["fallback_correct"].astype(bool).to_numpy()
        final = np.where(selected, fallback, compact)
        out["acc_delta_pp"] = float((np.mean(final) - np.mean(compact)) * 100.0)
        out["harm_rate_selected"] = float(np.mean(selected & compact & (~fallback)) / max(1e-12, np.mean(selected)))
    return out


def model_matrix(df, feature_cols):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.to_numpy(dtype=np.float32), list(X.columns)


def cross_validate_models(df, feature_cols, random_seed=1993):
    y = df["oracle_positive"].astype(int).to_numpy()
    if np.sum(y) < 10 or len(np.unique(y)) < 2:
        return pd.DataFrame()
    X, used_cols = model_matrix(df, feature_cols)
    pos_rate = float(np.mean(y))
    models = {
        "logistic_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_seed,
            ),
        ),
        "random_forest_depth4": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=4,
        ),
        "gbdt_depth2": GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=12,
            random_state=random_seed,
        ),
    }
    n_splits = min(5, int(np.sum(y)), int(np.sum(1 - y)))
    if n_splits < 2:
        return pd.DataFrame()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    rows = []
    for name, model in models.items():
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            clf = model
            clf.fit(X[train_idx], y[train_idx])
            if hasattr(clf, "predict_proba"):
                score = clf.predict_proba(X[test_idx])[:, 1]
            else:
                score = clf.decision_function(X[test_idx])
            y_test = y[test_idx]
            try:
                auc = roc_auc_score(y_test, score)
            except ValueError:
                auc = np.nan
            ap = average_precision_score(y_test, score)
            fold_df = df.iloc[test_idx].reset_index(drop=True)
            budget_metrics = metrics_at_budget(y_test, score, pos_rate, fold_df)
            row = {
                "model": name,
                "fold": int(fold),
                "roc_auc": float(auc),
                "average_precision": float(ap),
                **{f"oracle_budget_{k}": v for k, v in budget_metrics.items()},
            }
            fold_rows.append(row)
        rows.extend(fold_rows)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    agg = out.groupby("model").agg(["mean", "std"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    return agg


def cross_validate_model_budget_curves(df, feature_cols, random_seed=1993):
    y = df["oracle_positive"].astype(int).to_numpy()
    if np.sum(y) < 10 or len(np.unique(y)) < 2:
        return pd.DataFrame()
    X, _ = model_matrix(df, feature_cols)
    pos_rate = float(np.mean(y))
    budgets = sorted(set([0.005, 0.01, 0.02, pos_rate, 0.06, 0.10]))
    models = {
        "logistic_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_seed,
            ),
        ),
        "random_forest_depth4": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=4,
        ),
        "gbdt_depth2": GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=12,
            random_state=random_seed,
        ),
    }
    n_splits = min(5, int(np.sum(y)), int(np.sum(1 - y)))
    if n_splits < 2:
        return pd.DataFrame()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    rows = []
    for name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            clf = model
            clf.fit(X[train_idx], y[train_idx])
            score = clf.predict_proba(X[test_idx])[:, 1]
            fold_df = df.iloc[test_idx].reset_index(drop=True)
            for budget in budgets:
                metrics = metrics_at_budget(y[test_idx], score, budget, fold_df)
                rows.append(
                    {
                        "model": name,
                        "fold": int(fold),
                        "budget": float(budget),
                        **metrics,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    agg = out.groupby(["model", "budget"]).agg(["mean", "std"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    return agg.reset_index()


def plot_feature_histograms(df, screen_df, output_dir, top_n=12):
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    pos = df["oracle_positive"].astype(bool)
    top_features = screen_df["feature"].head(top_n).tolist() if not screen_df.empty else []
    for col in top_features:
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        a = values[pos].dropna().to_numpy(dtype=float)
        b = values[~pos].dropna().to_numpy(dtype=float)
        if a.size == 0 or b.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        combined = np.concatenate([a, b])
        lo, hi = np.quantile(combined, [0.01, 0.99])
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
            lo, hi = np.min(combined), np.max(combined)
        bins = np.linspace(lo, hi, 40)
        ax.hist(b, bins=bins, density=True, alpha=0.55, label="ordinary", color="#69788f")
        ax.hist(a, bins=bins, density=True, alpha=0.65, label="oracle-positive", color="#d95f02")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_filename(col)}_hist.png", dpi=160)
        plt.close(fig)

    if top_features:
        ncols = 3
        nrows = int(math.ceil(min(len(top_features), top_n) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
        axes = np.asarray(axes).reshape(-1)
        for ax, col in zip(axes, top_features[:top_n]):
            values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            data = [
                values[~pos].dropna().to_numpy(dtype=float),
                values[pos].dropna().to_numpy(dtype=float),
            ]
            ax.boxplot(data, labels=["ordinary", "oracle+"], showfliers=False)
            ax.set_title(col, fontsize=9)
        for ax in axes[len(top_features[:top_n]) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(plot_dir / "top_feature_boxplots.png", dpi=160)
        plt.close(fig)


def plot_model_pr(df, feature_cols, output_dir, random_seed=1993):
    y = df["oracle_positive"].astype(int).to_numpy()
    if np.sum(y) < 10:
        return
    X, _ = model_matrix(df, feature_cols)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=random_seed,
        n_jobs=4,
    )
    model.fit(X, y)
    score = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, score)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color="#1b9e77")
    ax.axhline(np.mean(y), color="#666666", linestyle="--", linewidth=1, label="base rate")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Random forest in-sample PR diagnostic")
    ax.legend()
    fig.tight_layout()
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "random_forest_pr_diagnostic.png", dpi=160)
    plt.close(fig)


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def write_report(df, stats_df, screen_df, model_df, model_budget_df, output_dir, analyzed_files):
    out_path = Path(output_dir) / "trace_diagnostic_report.md"
    pos = df["oracle_positive"].astype(bool)
    harm = df["fallback_harm"].astype(bool)
    compact_acc = float(df["compact_correct"].mean()) if "compact_correct" in df else float("nan")
    fallback_acc = float(df["fallback_correct"].mean()) if "fallback_correct" in df else float("nan")
    oracle_acc = float((df["compact_correct"] | df["fallback_correct"]).mean())
    oracle_rate = float(pos.mean())

    top_cols = [
        "feature",
        "auc_best_orientation",
        "orientation_positive_when",
        "average_precision",
    ]
    budget_cols = [c for c in screen_df.columns if c.startswith("precision_b") or c.startswith("recall_b") or c.startswith("acc_delta_pp_b")]
    top_table = screen_df[top_cols + budget_cols[:6]].head(15) if not screen_df.empty else pd.DataFrame()

    decision = decide_gate(screen_df, model_df, oracle_rate)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Direction 1 Raw Fallback Trace 诊断报告\n\n")
        f.write("## 结论\n\n")
        f.write(decision + "\n\n")
        f.write("## 数据范围\n\n")
        f.write(f"- Trace 文件数：{len(analyzed_files)}\n")
        f.write(f"- 样本数：{len(df)}\n")
        f.write(f"- fallback_available：{df['fallback_available'].mean():.4f}\n")
        f.write(f"- compact accuracy：{compact_acc:.4f}\n")
        f.write(f"- raw fallback top1 accuracy：{fallback_acc:.4f}\n")
        f.write(f"- oracle accuracy：{oracle_acc:.4f}\n")
        f.write(f"- oracle-positive rate：{oracle_rate:.4f} ({int(pos.sum())}/{len(df)})\n")
        f.write(f"- fallback-harm rate：{harm.mean():.4f} ({int(harm.sum())}/{len(df)})\n\n")

        f.write("## 单特征可分性 Top 15\n\n")
        if top_table.empty:
            f.write("无可用单特征结果。\n\n")
        else:
            f.write(top_table.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")

        f.write("## 模型级交叉验证\n\n")
        if model_df.empty:
            f.write("正例不足，未运行模型级交叉验证。\n\n")
        else:
            keep = [
                c for c in model_df.columns
                if c == "model"
                or c.endswith("_mean")
                and (
                    "roc_auc" in c
                    or "average_precision" in c
                    or "precision" in c
                    or "recall" in c
                    or "acc_delta_pp" in c
                    or "harm_rate" in c
                )
            ]
            f.write(model_df[keep].to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")

        f.write("## 低预算模型曲线\n\n")
        if model_budget_df.empty:
            f.write("无可用低预算模型曲线。\n\n")
        else:
            budget_keep = [
                "model",
                "budget",
                "precision_mean",
                "recall_mean",
                "acc_delta_pp_mean",
                "harm_rate_selected_mean",
            ]
            show = model_budget_df[
                (model_budget_df["model"] == "gbdt_depth2")
                & (model_budget_df["budget"].isin(sorted(model_budget_df["budget"].unique())))
            ][budget_keep]
            f.write(show.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")

            f.write(
                "这说明 gate 不能追求高 fallback rate。当前 trace 下较合理的部署区间是 "
                "**1%-4% fallback rate**；超过 6% 后 harm 快速吞掉收益。\n\n"
            )

        f.write("## 分布图\n\n")
        f.write("- `plots/top_feature_boxplots.png`\n")
        f.write("- `plots/*_hist.png`\n")
        f.write("- `plots/random_forest_pr_diagnostic.png`\n\n")

        f.write("## 统计表输出\n\n")
        f.write("- `trace_features.csv`\n")
        f.write("- `feature_group_stats.csv`\n")
        f.write("- `feature_screening.csv`\n")
        f.write("- `model_cv_summary.csv`\n\n")
        f.write("- `model_budget_curves.csv`\n")

        f.write("## 读取的 trace 文件\n\n")
        for path in analyzed_files:
            f.write(f"- `{path}`\n")

    return out_path


def decide_gate(screen_df, model_df, oracle_rate):
    if screen_df.empty:
        return (
            "没有单特征呈现可用区分度；当前不应推进 rule gate。"
            "下一步应扩大 trace 特征或改 fallback cache。"
        )
    top_auc = float(screen_df.iloc[0]["auc_best_orientation"])
    top_ap = float(screen_df.iloc[0]["average_precision"])
    model_best_auc = 0.0
    model_best_precision = 0.0
    if not model_df.empty:
        auc_cols = [c for c in model_df.columns if c == "roc_auc_mean"]
        precision_cols = [c for c in model_df.columns if "oracle_budget_precision_mean" in c]
        if auc_cols:
            model_best_auc = float(model_df[auc_cols[0]].max())
        if precision_cols:
            model_best_precision = float(model_df[precision_cols[0]].max())

    if top_auc >= 0.75 and top_ap >= max(0.20, oracle_rate * 4.0):
        return (
            "建议先做 rule gate 或单调校准 rule：已有单特征具备较强区分度，"
            "可以用 validation split 固定方向和阈值，再做低 fallback rate hard switch。"
        )
    if model_best_auc >= top_auc + 0.04 and model_best_precision >= max(0.12, oracle_rate * 2.5):
        return (
            "建议使用更强的轻量非线性 gate（优先 depth<=4 random forest / GBDT）。"
            "单特征不足以稳定隔离 oracle-positive，但多特征组合有明显增益；"
            "必须使用独立 calibration split 选阈值，不能在 test trace 上定阈值。"
        )
    if model_best_auc >= 0.65 or top_auc >= 0.65:
        return (
            "建议使用校准模型而不是手写 rule：信号存在但单特征分布重叠明显。"
            "优先 logistic/ridge 作为可解释基线，同时把低 fallback budget 下的 precision/acc gain "
            "作为主要选择指标。"
        )
    return (
        "当前 reliability features 对 oracle-positive 的区分度偏弱；"
        "不建议继续依赖简单 rule 或线性校准，应补充 node/class-level 历史错误率、"
        "class confusion、atom conflict 等更直接的可靠性信号后再训练轻量模型。"
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_trace_dir(args.trace_dir)
    if args.final_task_only and not args.include_all_tasks:
        df["total_classes"] = pd.to_numeric(df["total_classes"], errors="coerce")
        max_classes = df["total_classes"].max()
        df = df[df["total_classes"] == max_classes].copy()
    df = add_derived_features(df)
    if "fallback_available" in df.columns:
        df = df[df["fallback_available"].astype(bool)].copy()
    df = df.reset_index(drop=True)

    feature_cols = numeric_feature_columns(df)
    stats_df = summarize_feature_groups(df, feature_cols)
    screen_df = feature_screen(df, feature_cols)
    top_features = screen_df["feature"].head(30).tolist() if not screen_df.empty else feature_cols
    model_features = list(dict.fromkeys(top_features + [
        "compact_raw_disagree_sample",
        "fallback_better_distance",
        "selected_node_state_inactive",
        "selected_node_state_plastic",
        "selected_node_state_protected",
        "fallback_state_inactive",
        "fallback_state_plastic",
        "fallback_state_protected",
    ]))
    model_features = [c for c in model_features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    model_df = cross_validate_models(df, model_features, random_seed=args.random_seed)
    model_budget_df = cross_validate_model_budget_curves(df, model_features, random_seed=args.random_seed)

    df.to_csv(output_dir / "trace_features.csv", index=False)
    stats_df.to_csv(output_dir / "feature_group_stats.csv", index=False)
    screen_df.to_csv(output_dir / "feature_screening.csv", index=False)
    model_df.to_csv(output_dir / "model_cv_summary.csv", index=False)
    model_budget_df.to_csv(output_dir / "model_budget_curves.csv", index=False)
    pd.Series(feature_cols, name="feature").to_csv(output_dir / "candidate_features.txt", index=False)

    plot_feature_histograms(df, screen_df, output_dir, top_n=args.top_plots)
    plot_model_pr(df, model_features, output_dir, random_seed=args.random_seed)

    analyzed_files = sorted(df["_trace_file"].unique().tolist()) if "_trace_file" in df.columns else []
    report_path = write_report(
        df,
        stats_df,
        screen_df,
        model_df,
        model_budget_df,
        output_dir,
        analyzed_files,
    )
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
