import logging
import json
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from utils.inc_net import SimpleVitNetKNN
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.hc_soinn_classifier import HCSOINNClassifier
from utils.feature_cache import (
    feature_cache_classifier_only,
    feature_cache_classifier_torch_device,
    get_cached_feature_dataset,
    get_cached_feature_dataset_split,
    is_cached_feature_loader,
    log_feature_cache_loader,
)


num_workers = 8
batch_size = 128


class Learner(BaseLearner):
    """LifeTopoDict learner: SimpleCIL backbone with HC-SOINN + dictionary coding.

    Inherits the same frozen-backbone, prototype-only philosophy as
    ``simplecil_hc_soinn`` but enables ``use_dict_coding=True`` in the
    HC-SOINN classifier so that sparse dictionary coding, lifecycle
    management and residual-growth are activated during ``compress()``.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._feature_cache_classifier_only = feature_cache_classifier_only(args)
        self._classifier_device = (
            feature_cache_classifier_torch_device(args, self._device)
            if self._feature_cache_classifier_only
            else self._device
        )
        if self._feature_cache_classifier_only:
            self._network = None
            logging.info(
                "[FeatureCache] Classifier-only mode enabled: skipping backbone "
                "initialization for LifeTopoDict; classifier_device=%s",
                self._classifier_device,
            )
        else:
            self._network = SimpleVitNetKNN(args, True)

        # --- LifeTopoDict-specific parameters ---
        dict_sparse_k = args.get("dict_sparse_k", 5)
        dict_ridge_lambda = args.get("dict_ridge_lambda", 0.1)
        lifecycle_theta_support = args.get("lifecycle_theta_support", 0.5)
        lifecycle_theta_usage = args.get("lifecycle_theta_usage", 0.01)
        lifecycle_T_inactive = args.get("lifecycle_T_inactive", 3)
        lifecycle_min_alive_ratio = args.get("lifecycle_min_alive_ratio", 0.70)
        lifecycle_protect_old_topk = args.get("lifecycle_protect_old_topk", True)
        lifecycle_node_inactive_threshold = args.get("lifecycle_node_inactive_threshold", 1.0)
        dict_theta_residual = args.get("dict_theta_residual", 0.3)
        dict_max_growth_per_task = args.get("dict_max_growth_per_task", 0)
        drop_node_raw_after_dict_materialize = args.get(
            "drop_node_raw_after_dict_materialize", True
        )
        use_raw_fallback_gate = args.get("use_raw_fallback_gate", False)
        raw_fallback_per_class = args.get("raw_fallback_per_class", 0)
        raw_fallback_select_by = args.get("raw_fallback_select_by", "residual")
        fallback_gate_type = args.get("fallback_gate_type", "rule")
        fallback_gate_margin_threshold = args.get("fallback_gate_margin_threshold", 0.05)
        fallback_gate_residual_threshold = args.get("fallback_gate_residual_threshold", 0.08)
        fallback_gate_random_rate = args.get("fallback_gate_random_rate", 0.10)
        fallback_memory_accounting = args.get("fallback_memory_accounting", True)
        fallback_calibration_samples_per_class = args.get("fallback_calibration_samples_per_class", 0)
        fallback_calibration_cumulative = args.get("fallback_calibration_cumulative", True)
        fallback_gate_min_positives = args.get("fallback_gate_min_positives", 5)
        fallback_gate_min_calibration_gain = args.get("fallback_gate_min_calibration_gain", 0.001)
        fallback_gate_max_rate = args.get("fallback_gate_max_rate", 1.0)
        fallback_gate_tree_estimators = args.get("fallback_gate_tree_estimators", 160)
        fallback_gate_tree_max_depth = args.get("fallback_gate_tree_max_depth", 2)
        fallback_gate_tree_min_samples_leaf = args.get("fallback_gate_tree_min_samples_leaf", 12)
        fallback_gate_threshold_strategy = args.get("fallback_gate_threshold_strategy", "max_net")
        fallback_gate_min_precision = args.get("fallback_gate_min_precision", 0.0)
        fallback_gate_min_selected = args.get("fallback_gate_min_selected", 0)
        fallback_gate_min_net_count = args.get("fallback_gate_min_net_count", 0)
        fallback_candidate_rule = args.get("fallback_candidate_rule", "none")
        fallback_gate_budget_rates = args.get("fallback_gate_budget_rates", "0.002,0.005,0.01,0.02,0.04")
        fallback_gate_neutral_weight = args.get("fallback_gate_neutral_weight", 0.25)
        fallback_pair_table_mode = args.get("fallback_pair_table_mode", "none")
        fallback_pair_table_smoothing = args.get("fallback_pair_table_smoothing", 10.0)
        fallback_pair_table_lambda = args.get("fallback_pair_table_lambda", 0.5)
        fallback_node_table_lambda = args.get("fallback_node_table_lambda", 0.5)
        use_raw_auxiliary_nodes = args.get("use_raw_auxiliary_nodes", False)
        raw_auxiliary_penalty = args.get("raw_auxiliary_penalty", 0.0)
        raw_auxiliary_scope = args.get("raw_auxiliary_scope", "all")
        enable_prediction_trace = args.get("enable_prediction_trace", False)
        trace_split = args.get("trace_split", "test")
        use_atom_conflict_gate = args.get("use_atom_conflict_gate", False)
        atom_conflict_metric = args.get("atom_conflict_metric", "pmi")
        atom_conflict_target = args.get("atom_conflict_target", "all_errors")
        atom_conflict_topk = args.get("atom_conflict_topk", 80)
        atom_gate_strength = args.get("atom_gate_strength", 0.03)
        atom_gate_min_pair_support = args.get("atom_gate_min_pair_support", 3)
        atom_gate_smoothing = args.get("atom_gate_smoothing", 5.0)
        atom_gate_calibration_samples_per_class = args.get(
            "atom_gate_calibration_samples_per_class", 0
        )
        atom_gate_calibration_cumulative = args.get("atom_gate_calibration_cumulative", True)
        use_score_bias_calibration = args.get("use_score_bias_calibration", False)
        score_bias_calibration_samples_per_class = args.get(
            "score_bias_calibration_samples_per_class", 0
        )
        score_bias_calibration_cumulative = args.get("score_bias_calibration_cumulative", True)
        score_bias_grid = args.get(
            "score_bias_grid",
            "-0.03,-0.02,-0.015,-0.01,-0.005,0,0.005,0.01,0.015,0.02,0.03",
        )
        score_bias_min_gain = args.get("score_bias_min_gain", 0.0)
        use_node_residual_penalty = args.get("use_node_residual_penalty", False)
        node_residual_penalty_strength = args.get("node_residual_penalty_strength", 0.0)
        node_residual_penalty_mode = args.get("node_residual_penalty_mode", "linear")
        use_class_score_normalization = args.get("use_class_score_normalization", False)
        class_score_norm_mode = args.get("class_score_norm_mode", "affine")
        class_score_norm_strength = args.get("class_score_norm_strength", 1.0)
        class_score_norm_min_scale = args.get("class_score_norm_min_scale", 0.01)
        use_node_density_scoring = args.get("use_node_density_scoring", False)
        node_density_scoring_strength = args.get("node_density_scoring_strength", 0.0)
        node_density_scoring_mode = args.get("node_density_scoring_mode", "log_count")
        node_density_scoring_clip = args.get("node_density_scoring_clip", 2.0)

        # --- Ablation switches. Growth and additive edge-aware scoring are
        # kept for historical reproducibility only; future experiments keep
        # them disabled.
        use_dictionary_growth = args.get("use_dictionary_growth", False)
        use_lifecycle = args.get("use_lifecycle", True)
        use_protected_gate = args.get("use_protected_gate", True)
        use_edge_age_persistence = args.get("use_edge_age_persistence", True)
        use_edge_aware_scoring = args.get("use_edge_aware_scoring", False)
        edge_score_gamma = args.get("edge_score_gamma", 0.1)
        edge_score_eta = args.get("edge_score_eta", 0.05)

        self.hc_soinn = HCSOINNClassifier(
            max_prototypes_per_class=args.get("hcsoinn_max_proto_per_class", 20),
            alpha=args.get("hcsoinn_alpha", 0.5),
            tau_merge=args.get("hcsoinn_tau_merge", 0.2),
            tau_reject=args.get("hcsoinn_tau_reject", 2.0),
            linkage_method=args.get("hcsoinn_linkage", "average"),
            distance_metric=args.get("hcsoinn_distance", "cosine"),
            use_soinn_refinement=args.get("hcsoinn_use_soinn_refinement", True),
            soinn_ad=args.get("hcsoinn_soinn_ad", 20),
            soinn_lam=args.get("hcsoinn_soinn_lam", 20),
            soinn_threshold_scale=args.get("hcsoinn_soinn_threshold_scale", 0.5),
            soinn_max_iter=args.get("hcsoinn_soinn_max_iter", 3),
            soinn_max_degree_for_removal=args.get("hcsoinn_soinn_max_degree_for_removal", 1),
            # Lifecycle parameters
            lifecycle_theta_support=lifecycle_theta_support,
            lifecycle_theta_usage=lifecycle_theta_usage,
            lifecycle_T_inactive=lifecycle_T_inactive,
            lifecycle_min_alive_ratio=lifecycle_min_alive_ratio,
            lifecycle_protect_old_topk=lifecycle_protect_old_topk,
            lifecycle_node_inactive_threshold=lifecycle_node_inactive_threshold,
            # --- Enable dictionary coding ---
            use_dict_coding=args.get("use_dict_coding", True),
            dict_sparse_k=dict_sparse_k,
            dict_ridge_lambda=dict_ridge_lambda,
            drop_node_raw_after_dict_materialize=drop_node_raw_after_dict_materialize,
            # --- Dictionary growth parameters ---
            theta_residual=dict_theta_residual,
            max_growth_per_task=dict_max_growth_per_task,
            # --- Direction 1: raw-node fallback gate ---
            use_raw_fallback_gate=use_raw_fallback_gate,
            raw_fallback_per_class=raw_fallback_per_class,
            raw_fallback_select_by=raw_fallback_select_by,
            fallback_gate_type=fallback_gate_type,
            fallback_gate_margin_threshold=fallback_gate_margin_threshold,
            fallback_gate_residual_threshold=fallback_gate_residual_threshold,
            fallback_gate_random_rate=fallback_gate_random_rate,
            fallback_memory_accounting=fallback_memory_accounting,
            fallback_calibration_samples_per_class=fallback_calibration_samples_per_class,
            fallback_gate_min_positives=fallback_gate_min_positives,
            fallback_gate_min_calibration_gain=fallback_gate_min_calibration_gain,
            fallback_gate_max_rate=fallback_gate_max_rate,
            fallback_gate_tree_estimators=fallback_gate_tree_estimators,
            fallback_gate_tree_max_depth=fallback_gate_tree_max_depth,
            fallback_gate_tree_min_samples_leaf=fallback_gate_tree_min_samples_leaf,
            fallback_gate_threshold_strategy=fallback_gate_threshold_strategy,
            fallback_gate_min_precision=fallback_gate_min_precision,
            fallback_gate_min_selected=fallback_gate_min_selected,
            fallback_gate_min_net_count=fallback_gate_min_net_count,
            fallback_candidate_rule=fallback_candidate_rule,
            fallback_gate_budget_rates=fallback_gate_budget_rates,
            fallback_gate_neutral_weight=fallback_gate_neutral_weight,
            fallback_pair_table_mode=fallback_pair_table_mode,
            fallback_pair_table_smoothing=fallback_pair_table_smoothing,
            fallback_pair_table_lambda=fallback_pair_table_lambda,
            fallback_node_table_lambda=fallback_node_table_lambda,
            use_raw_auxiliary_nodes=use_raw_auxiliary_nodes,
            raw_auxiliary_penalty=raw_auxiliary_penalty,
            raw_auxiliary_scope=raw_auxiliary_scope,
            enable_prediction_trace=enable_prediction_trace,
            trace_split=trace_split,
            fallback_random_seed=args.get("fallback_random_seed", args.get("seed", 0)),
            use_atom_conflict_gate=use_atom_conflict_gate,
            atom_conflict_metric=atom_conflict_metric,
            atom_conflict_target=atom_conflict_target,
            atom_conflict_topk=atom_conflict_topk,
            atom_gate_strength=atom_gate_strength,
            atom_gate_min_pair_support=atom_gate_min_pair_support,
            atom_gate_smoothing=atom_gate_smoothing,
            atom_gate_calibration_samples_per_class=atom_gate_calibration_samples_per_class,
            use_score_bias_calibration=use_score_bias_calibration,
            score_bias_calibration_samples_per_class=score_bias_calibration_samples_per_class,
            score_bias_grid=score_bias_grid,
            score_bias_min_gain=score_bias_min_gain,
            use_node_residual_penalty=use_node_residual_penalty,
            node_residual_penalty_strength=node_residual_penalty_strength,
            node_residual_penalty_mode=node_residual_penalty_mode,
            use_class_score_normalization=use_class_score_normalization,
            class_score_norm_mode=class_score_norm_mode,
            class_score_norm_strength=class_score_norm_strength,
            class_score_norm_min_scale=class_score_norm_min_scale,
            use_node_density_scoring=use_node_density_scoring,
            node_density_scoring_strength=node_density_scoring_strength,
            node_density_scoring_mode=node_density_scoring_mode,
            node_density_scoring_clip=node_density_scoring_clip,
        )

        # --- P0-4: Apply ablation switches (override defaults from HCSOINNClassifier) ---
        self.hc_soinn.use_dictionary_growth = bool(use_dictionary_growth)
        self.hc_soinn.use_lifecycle = bool(use_lifecycle)
        self.hc_soinn.use_protected_gate = bool(use_protected_gate)
        self.hc_soinn.use_edge_age_persistence = bool(use_edge_age_persistence)
        self.hc_soinn.use_edge_aware_scoring = bool(use_edge_aware_scoring)
        self.hc_soinn.edge_score_gamma = float(edge_score_gamma)
        self.hc_soinn.edge_score_eta = float(edge_score_eta)
        self._hc_soinn_compressed_for_task = False
        self._fallback_calibration_cumulative = bool(fallback_calibration_cumulative)
        self._fallback_calibration_datasets = []
        self._atom_gate_calibration_cumulative = bool(atom_gate_calibration_cumulative)
        self._atom_gate_calibration_datasets = []
        self._score_bias_calibration_cumulative = bool(score_bias_calibration_cumulative)
        self._score_bias_calibration_datasets = []
        self._prediction_trace_output_dir = args.get("prediction_trace_output_dir", None)
        self._prediction_trace_dump_all_tasks = bool(
            args.get("prediction_trace_dump_all_tasks", True)
        )

        logging.info(
            f"[LifeTopoDict] Initialized with dict_sparse_k={dict_sparse_k}, "
            f"dict_ridge_lambda={dict_ridge_lambda}, "
            f"lifecycle_theta_support={lifecycle_theta_support}, "
            f"lifecycle_theta_usage={lifecycle_theta_usage}, "
            f"lifecycle_T_inactive={lifecycle_T_inactive}, "
            f"lifecycle_min_alive_ratio={lifecycle_min_alive_ratio}, "
            f"lifecycle_protect_old_topk={lifecycle_protect_old_topk}, "
            f"lifecycle_node_inactive_threshold={lifecycle_node_inactive_threshold}, "
            f"drop_node_raw_after_dict_materialize={drop_node_raw_after_dict_materialize}, "
            f"dict_theta_residual={dict_theta_residual}, "
            f"dict_max_growth_per_task={dict_max_growth_per_task}, "
            f"use_dictionary_growth={use_dictionary_growth}, "
            f"use_lifecycle={use_lifecycle}, "
            f"use_protected_gate={use_protected_gate}, "
            f"use_edge_age_persistence={use_edge_age_persistence}, "
            f"use_edge_aware_scoring={use_edge_aware_scoring}, "
            f"edge_score_gamma={edge_score_gamma}, "
            f"edge_score_eta={edge_score_eta}, "
            f"use_raw_fallback_gate={use_raw_fallback_gate}, "
            f"raw_fallback_per_class={raw_fallback_per_class}, "
            f"raw_fallback_select_by={raw_fallback_select_by}, "
            f"fallback_gate_type={fallback_gate_type}, "
            f"fallback_gate_margin_threshold={fallback_gate_margin_threshold}, "
            f"fallback_gate_residual_threshold={fallback_gate_residual_threshold}, "
            f"fallback_gate_random_rate={fallback_gate_random_rate}, "
            f"fallback_memory_accounting={fallback_memory_accounting}, "
            f"fallback_calibration_samples_per_class={fallback_calibration_samples_per_class}, "
            f"fallback_calibration_cumulative={fallback_calibration_cumulative}, "
            f"fallback_gate_min_positives={fallback_gate_min_positives}, "
            f"fallback_gate_min_calibration_gain={fallback_gate_min_calibration_gain}, "
            f"fallback_gate_max_rate={fallback_gate_max_rate}, "
            f"fallback_gate_tree_estimators={fallback_gate_tree_estimators}, "
            f"fallback_gate_tree_max_depth={fallback_gate_tree_max_depth}, "
            f"fallback_gate_tree_min_samples_leaf={fallback_gate_tree_min_samples_leaf}, "
            f"fallback_gate_threshold_strategy={fallback_gate_threshold_strategy}, "
            f"fallback_gate_min_precision={fallback_gate_min_precision}, "
            f"fallback_gate_min_selected={fallback_gate_min_selected}, "
            f"fallback_gate_min_net_count={fallback_gate_min_net_count}, "
            f"fallback_candidate_rule={fallback_candidate_rule}, "
            f"fallback_gate_budget_rates={fallback_gate_budget_rates}, "
            f"fallback_gate_neutral_weight={fallback_gate_neutral_weight}, "
            f"fallback_pair_table_mode={fallback_pair_table_mode}, "
            f"fallback_pair_table_smoothing={fallback_pair_table_smoothing}, "
            f"fallback_pair_table_lambda={fallback_pair_table_lambda}, "
            f"fallback_node_table_lambda={fallback_node_table_lambda}, "
            f"use_raw_auxiliary_nodes={use_raw_auxiliary_nodes}, "
            f"raw_auxiliary_penalty={raw_auxiliary_penalty}, "
            f"raw_auxiliary_scope={raw_auxiliary_scope}, "
            f"use_atom_conflict_gate={use_atom_conflict_gate}, "
            f"atom_conflict_metric={atom_conflict_metric}, "
            f"atom_conflict_target={atom_conflict_target}, "
            f"atom_conflict_topk={atom_conflict_topk}, "
            f"atom_gate_strength={atom_gate_strength}, "
            f"atom_gate_min_pair_support={atom_gate_min_pair_support}, "
            f"atom_gate_smoothing={atom_gate_smoothing}, "
            f"atom_gate_calibration_samples_per_class={atom_gate_calibration_samples_per_class}, "
            f"atom_gate_calibration_cumulative={atom_gate_calibration_cumulative}, "
            f"use_score_bias_calibration={use_score_bias_calibration}, "
            f"score_bias_calibration_samples_per_class={score_bias_calibration_samples_per_class}, "
            f"score_bias_calibration_cumulative={score_bias_calibration_cumulative}, "
            f"score_bias_grid={score_bias_grid}, "
            f"score_bias_min_gain={score_bias_min_gain}, "
            f"use_node_residual_penalty={use_node_residual_penalty}, "
            f"node_residual_penalty_strength={node_residual_penalty_strength}, "
            f"node_residual_penalty_mode={node_residual_penalty_mode}, "
            f"use_class_score_normalization={use_class_score_normalization}, "
            f"class_score_norm_mode={class_score_norm_mode}, "
            f"class_score_norm_strength={class_score_norm_strength}, "
            f"class_score_norm_min_scale={class_score_norm_min_scale}, "
            f"use_node_density_scoring={use_node_density_scoring}, "
            f"node_density_scoring_strength={node_density_scoring_strength}, "
            f"node_density_scoring_mode={node_density_scoring_mode}, "
            f"node_density_scoring_clip={node_density_scoring_clip}, "
            f"enable_prediction_trace={enable_prediction_trace}, "
            f"trace_split={trace_split}, "
            f"prediction_trace_output_dir={self._prediction_trace_output_dir}, "
            f"prediction_trace_dump_all_tasks={self._prediction_trace_dump_all_tasks}"
        )

    # ------------------------------------------------------------------ #
    # Task boundary: compress triggers dict coding, growth, lifecycle     #
    # ------------------------------------------------------------------ #
    def _compress_task_boundary(self):
        if self._hc_soinn_compressed_for_task:
            return
        try:
            if hasattr(self.hc_soinn, "set_task_boundary"):
                self.hc_soinn.set_task_boundary(self._known_classes)
            self.hc_soinn.compress()
            self._hc_soinn_compressed_for_task = True
        except Exception as e:
            logging.error(f"[LifeTopoDict] compress error: {e}", exc_info=True)

    def after_task(self):
        if not self._hc_soinn_compressed_for_task:
            self._compress_task_boundary()

        # Output memory report after compress
        diag = self.hc_soinn.compute_diagnostics()
        mem = diag.get('memory', {})
        edge_stats = diag.get('edge_score_stats', {})
        logging.info(
            f"[LifeTopoDict] Memory: compact={mem.get('compact_deployable_mb', 0):.4f} MB, "
            f"actual={mem.get('actual_implementation_mb', 0):.4f} MB, "
            f"atoms={mem.get('atoms_mb', 0):.4f} MB, "
            f"coeffs={mem.get('coefficients_mb', 0):.4f} MB, "
            f"edges={mem.get('edges_mb', 0):.4f} MB, "
            f"edge_rel={mem.get('edge_reliability_mb', 0):.4f} MB, "
            f"fallback={mem.get('raw_fallback_total_mb', 0):.4f} MB, "
            f"fallback_gate={mem.get('raw_fallback_gate_model_mb', 0):.4f} MB, "
            f"atom_gate={mem.get('atom_conflict_gate_model_mb', 0):.4f} MB, "
            f"score_bias={mem.get('score_bias_model_mb', 0):.4f} MB, "
            f"residual_penalty={mem.get('node_residual_penalty_model_mb', 0):.4f} MB, "
            f"raw_aux={mem.get('raw_auxiliary_model_mb', 0):.4f} MB, "
            f"class_score_norm={mem.get('class_score_normalization_model_mb', 0):.4f} MB, "
            f"node_density={mem.get('node_density_scoring_model_mb', 0):.4f} MB, "
            f"caches={mem.get('caches_mb', 0):.4f} MB, "
            f"buffers={mem.get('buffers_mb', 0):.4f} MB, "
            f"frozen={mem.get('frozen_mb', 0):.4f} MB"
        )
        fallback_stats = diag.get('raw_fallback_stats', {})
        if fallback_stats:
            logging.info(
                f"[LifeTopoDict] Raw fallback: enabled={int(fallback_stats.get('fallback_enabled', 0))}, "
                f"cache_nodes={int(fallback_stats.get('fallback_cache_nodes', 0))}, "
                f"cache_classes={int(fallback_stats.get('fallback_cache_classes', 0))}, "
                f"rate={fallback_stats.get('fallback_rate', 0):.4f}, "
                f"available={fallback_stats.get('fallback_available_rate', 0):.4f}, "
                f"change={fallback_stats.get('fallback_prediction_change_rate', 0):.4f}, "
                f"disagree={fallback_stats.get('compact_raw_disagreement', 0):.4f}, "
                f"compact_acc={fallback_stats.get('compact_accuracy', 0):.4f}, "
                f"fallback_acc={fallback_stats.get('fallback_accuracy', 0):.4f}, "
                f"final_acc={fallback_stats.get('final_accuracy', 0):.4f}, "
                f"used_compact_acc={fallback_stats.get('compact_correct_when_used', 0):.4f}, "
                f"used_fallback_acc={fallback_stats.get('fallback_correct_when_used', 0):.4f}, "
                f"oracle_improvable={fallback_stats.get('oracle_improvable_rate', 0):.4f}, "
                f"benefit={fallback_stats.get('benefit_rate', 0):.4f}, "
                f"harm={fallback_stats.get('harm_rate', 0):.4f}, "
                f"net={fallback_stats.get('net_gain_rate', 0):.4f}, "
                f"benefit_sel={fallback_stats.get('benefit_selected', 0):.0f}, "
                f"harm_sel={fallback_stats.get('harm_selected', 0):.0f}, "
                f"precision={fallback_stats.get('utility_precision', 0):.4f}, "
                f"benefit_recall={fallback_stats.get('benefit_recall', 0):.4f}, "
                f"margin_mean={fallback_stats.get('trace_margin_mean', 0):.4f}, "
                f"residual_mean={fallback_stats.get('trace_residual_mean', 0):.4f}, "
                f"used_margin_mean={fallback_stats.get('fallback_margin_mean', 0):.4f}, "
                f"used_residual_mean={fallback_stats.get('fallback_residual_mean', 0):.4f}"
            )
        raw_auxiliary_stats = diag.get('raw_auxiliary_stats', {})
        if raw_auxiliary_stats:
            logging.info(
                f"[LifeTopoDict] Raw auxiliary: enabled={int(raw_auxiliary_stats.get('raw_auxiliary_enabled', 0))}, "
                f"penalty={raw_auxiliary_stats.get('raw_auxiliary_penalty', 0):.4f}, "
                f"scope={raw_auxiliary_stats.get('raw_auxiliary_scope', 'all')}, "
                f"cache_nodes={int(raw_auxiliary_stats.get('raw_auxiliary_cache_nodes', 0))}, "
                f"cache_classes={int(raw_auxiliary_stats.get('raw_auxiliary_cache_classes', 0))}, "
                f"available={raw_auxiliary_stats.get('raw_auxiliary_available_rate', 0):.4f}, "
                f"change={raw_auxiliary_stats.get('raw_auxiliary_prediction_change_rate', 0):.4f}, "
                f"compact_acc={raw_auxiliary_stats.get('raw_auxiliary_compact_accuracy', 0):.4f}, "
                f"final_acc={raw_auxiliary_stats.get('raw_auxiliary_final_accuracy', 0):.4f}, "
                f"benefit={raw_auxiliary_stats.get('raw_auxiliary_benefit_selected', 0):.0f}, "
                f"harm={raw_auxiliary_stats.get('raw_auxiliary_harm_selected', 0):.0f}, "
                f"net={raw_auxiliary_stats.get('raw_auxiliary_net_gain_rate', 0):.4f}"
            )
        fallback_gate_stats = diag.get('raw_fallback_gate_stats', {})
        if fallback_gate_stats:
            logging.info(
                f"[LifeTopoDict] Raw fallback gate fit: samples={int(fallback_gate_stats.get('samples', 0))}, "
                f"positives={int(fallback_gate_stats.get('positives', 0))}, "
                f"pos_rate={fallback_gate_stats.get('positive_rate', 0):.4f}, "
                f"calib_acc={fallback_gate_stats.get('calibration_accuracy', 0):.4f}, "
                f"compact_acc={fallback_gate_stats.get('compact_accuracy', 0):.4f}, "
                f"fallback_acc={fallback_gate_stats.get('fallback_accuracy', 0):.4f}, "
                f"oracle_acc={fallback_gate_stats.get('oracle_accuracy', 0):.4f}, "
                f"gain={fallback_gate_stats.get('calibration_gain', 0):.4f}, "
                f"net={fallback_gate_stats.get('calibration_net_gain', 0):.4f}, "
                f"fallback_rate={fallback_gate_stats.get('calibration_fallback_rate', 0):.4f}, "
                f"benefit_sel={fallback_gate_stats.get('calibration_benefit_selected', 0):.0f}, "
                f"harm_sel={fallback_gate_stats.get('calibration_harm_selected', 0):.0f}, "
                f"precision={fallback_gate_stats.get('calibration_utility_precision', 0):.4f}, "
                f"candidate_rate={fallback_gate_stats.get('candidate_rate', 0):.4f}, "
                f"threshold={fallback_gate_stats.get('threshold', 0):.6f}, "
                f"weight_l2={fallback_gate_stats.get('weight_l2', 0):.4f}, "
                f"disabled={int(fallback_gate_stats.get('gate_disabled', 0))}"
            )
        atom_gate_stats = diag.get('atom_conflict_gate_stats', {})
        if atom_gate_stats:
            logging.info(
                f"[LifeTopoDict] Atom conflict gate fit: enabled={int(atom_gate_stats.get('enabled', 0))}, "
                f"samples={int(atom_gate_stats.get('samples', 0))}, "
                f"errors={int(atom_gate_stats.get('errors', 0))}, "
                f"global_error={atom_gate_stats.get('global_error_rate', 0):.4f}, "
                f"pairs={int(atom_gate_stats.get('pair_count', 0))}, "
                f"deployed={int(atom_gate_stats.get('deployed_pairs', 0))}, "
                f"metric={atom_gate_stats.get('metric', '')}, "
                f"target={atom_gate_stats.get('target', '')}, "
                f"strength={atom_gate_stats.get('strength', 0):.4f}, "
                f"calib_acc={atom_gate_stats.get('calibration_accuracy', 0):.4f}, "
                f"compact_acc={atom_gate_stats.get('compact_accuracy', 0):.4f}, "
                f"gain={atom_gate_stats.get('calibration_gain', 0):.4f}, "
                f"gate_rate={atom_gate_stats.get('calibration_gate_rate', 0):.4f}, "
                f"benefit_sel={atom_gate_stats.get('calibration_benefit_selected', 0):.0f}, "
                f"harm_sel={atom_gate_stats.get('calibration_harm_selected', 0):.0f}, "
                f"disabled={int(atom_gate_stats.get('gate_disabled', 0))}"
            )
        class_score_norm_stats = diag.get('class_score_normalization_stats', {})
        if class_score_norm_stats:
            logging.info(
                f"[LifeTopoDict] Class score norm: enabled={int(class_score_norm_stats.get('enabled', 0))}, "
                f"stored={int(class_score_norm_stats.get('stored_classes', 0))}, "
                f"fitted={int(class_score_norm_stats.get('fitted_classes', 0))}, "
                f"samples={int(class_score_norm_stats.get('samples', 0))}, "
                f"ref_center={class_score_norm_stats.get('ref_center', 0):.6f}, "
                f"ref_scale={class_score_norm_stats.get('ref_scale', 0):.6f}, "
                f"strength={class_score_norm_stats.get('strength', 0):.3f}"
            )
        if edge_stats:
            logging.info(
                f"[LifeTopoDict] Edge scoring: use_rate={edge_stats.get('edge_use_rate', 0):.4f}, "
                f"class_use_rate={edge_stats.get('edge_class_use_rate', 0):.4f}, "
                f"margin_contribution={edge_stats.get('edge_margin_contribution', 0):.4f}, "
                f"risk_penalty={edge_stats.get('edge_risk_penalty', 0):.4f}, "
                f"avg_adjustment={edge_stats.get('edge_score_adjustment', 0):.4f}"
            )

        self._known_classes = self._total_classes
        self._hc_soinn_compressed_for_task = False

    # ------------------------------------------------------------------ #
    # Feature extraction for HC-SOINN                                     #
    # ------------------------------------------------------------------ #
    def _extract_class_features(self, loader, model):
        feats, lbs = [], []
        from_cache = is_cached_feature_loader(loader)
        if not from_cache and model is None:
            raise RuntimeError(
                "Backbone is skipped, but train_loader_for_hc is not backed by cached features."
            )
        if model is not None:
            model.eval()
        with torch.no_grad():
            for _, data, label in loader:
                if from_cache:
                    emb = data.float()
                else:
                    data = data.to(self._device)
                    if isinstance(model, nn.DataParallel):
                        emb = model.module.extract_vector(data)
                    else:
                        emb = model.extract_vector(data)
                feats.append(emb.cpu())
                lbs.append(label.cpu())

        if len(feats) == 0:
            return None
        feats = torch.cat(feats, dim=0).numpy()
        lbs = torch.cat(lbs, dim=0).numpy()
        self.hc_soinn.add_features(feats, lbs)

        proto_info = self.hc_soinn.prototypes_per_class()
        logging.info(f"[LifeTopoDict] prototypes per class: {proto_info}")
        return feats, lbs

    # ------------------------------------------------------------------ #
    # Incremental training                                                #
    # ------------------------------------------------------------------ #
    def incremental_train(self, data_manager):
        logging.info(
            f"[LifeTopoDict] Starting incremental_train: cur_task={self._cur_task}, "
            f"known_classes={self._known_classes}"
        )
        self._hc_soinn_compressed_for_task = False
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            f"[LifeTopoDict] After update: cur_task={self._cur_task}, "
            f"total_classes={self._total_classes}, known_classes={self._known_classes}"
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        fallback_calibration_loader = None
        atom_gate_calibration_loader = None
        score_bias_calibration_loader = None

        test_dataset = get_cached_feature_dataset(
            self.args,
            data_manager,
            np.arange(0, self._total_classes),
            source="test",
        )
        if test_dataset is None:
            test_dataset = data_manager.get_dataset(
                np.arange(0, self._total_classes), source="test", mode="test"
            )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        train_dataset_for_hc = get_cached_feature_dataset(
            self.args,
            data_manager,
            np.arange(self._known_classes, self._total_classes),
            source="train",
        )
        learned_gate_types = (
            "ridge", "logistic", "gbdt", "gbdt_depth2", "random_forest", "rf", "rf_depth4",
            "candidate_rule", "delta_ridge", "delta_logistic", "benefit_harm_logistic",
            "dual_logistic", "dual_logistic_lcb", "utility_tree", "utility_gbdt",
            "utility_table", "delta_ridge_pair",
        )
        learned_fallback_gate = (
            bool(self.hc_soinn.use_raw_fallback_gate)
            and str(getattr(self.hc_soinn, "fallback_gate_type", "rule")).lower() in learned_gate_types
            and int(getattr(self.hc_soinn, "fallback_calibration_samples_per_class", 0)) > 0
        )
        learned_atom_gate = (
            bool(getattr(self.hc_soinn, "use_atom_conflict_gate", False))
            and int(getattr(self.hc_soinn, "atom_gate_calibration_samples_per_class", 0)) > 0
        )
        learned_score_bias = (
            bool(getattr(self.hc_soinn, "use_score_bias_calibration", False))
            and int(getattr(self.hc_soinn, "score_bias_calibration_samples_per_class", 0)) > 0
        )
        fallback_calibration_samples_per_class = int(
            getattr(self.hc_soinn, "fallback_calibration_samples_per_class", 0)
        )
        atom_gate_calibration_samples_per_class = int(
            getattr(self.hc_soinn, "atom_gate_calibration_samples_per_class", 0)
        )
        score_bias_calibration_samples_per_class = int(
            getattr(self.hc_soinn, "score_bias_calibration_samples_per_class", 0)
        )
        calibration_samples_per_class = max(
            fallback_calibration_samples_per_class if learned_fallback_gate else 0,
            atom_gate_calibration_samples_per_class if learned_atom_gate else 0,
            score_bias_calibration_samples_per_class if learned_score_bias else 0,
        )
        if learned_fallback_gate or learned_atom_gate or learned_score_bias:
            if train_dataset_for_hc is not None:
                train_dataset_for_hc, calibration_dataset = get_cached_feature_dataset_split(
                    self.args,
                    data_manager,
                    np.arange(self._known_classes, self._total_classes),
                    source="train",
                    val_samples_per_class=calibration_samples_per_class,
                )
            else:
                train_dataset_for_hc, calibration_dataset = data_manager.get_dataset_with_split(
                    np.arange(self._known_classes, self._total_classes),
                    source="train",
                    mode="test",
                    val_samples_per_class=calibration_samples_per_class,
                )
            if calibration_dataset is not None and learned_fallback_gate:
                if self._fallback_calibration_cumulative:
                    self._fallback_calibration_datasets.append(calibration_dataset)
                    if len(self._fallback_calibration_datasets) == 1:
                        fallback_calibration_dataset = self._fallback_calibration_datasets[0]
                    else:
                        fallback_calibration_dataset = ConcatDataset(
                            list(self._fallback_calibration_datasets)
                        )
                else:
                    fallback_calibration_dataset = calibration_dataset
                fallback_calibration_loader = DataLoader(
                    fallback_calibration_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
            if calibration_dataset is not None and learned_atom_gate:
                if self._atom_gate_calibration_cumulative:
                    self._atom_gate_calibration_datasets.append(calibration_dataset)
                    if len(self._atom_gate_calibration_datasets) == 1:
                        atom_gate_calibration_dataset = self._atom_gate_calibration_datasets[0]
                    else:
                        atom_gate_calibration_dataset = ConcatDataset(
                            list(self._atom_gate_calibration_datasets)
                        )
                else:
                    atom_gate_calibration_dataset = calibration_dataset
                atom_gate_calibration_loader = DataLoader(
                    atom_gate_calibration_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
            if calibration_dataset is not None and learned_score_bias:
                if self._score_bias_calibration_cumulative:
                    self._score_bias_calibration_datasets.append(calibration_dataset)
                    if len(self._score_bias_calibration_datasets) == 1:
                        score_bias_calibration_dataset = self._score_bias_calibration_datasets[0]
                    else:
                        score_bias_calibration_dataset = ConcatDataset(
                            list(self._score_bias_calibration_datasets)
                        )
                else:
                    score_bias_calibration_dataset = calibration_dataset
                score_bias_calibration_loader = DataLoader(
                    score_bias_calibration_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
        else:
            calibration_dataset = None
        if train_dataset_for_hc is None:
            train_dataset_for_hc = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="test",
            )
        self.train_loader_for_hc = DataLoader(
            train_dataset_for_hc,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        log_feature_cache_loader(self.train_loader_for_hc, "train_loader_for_hc")
        log_feature_cache_loader(self.test_loader, "test_loader")
        if fallback_calibration_loader is not None:
            log_feature_cache_loader(fallback_calibration_loader, "fallback_calibration_loader")
        if atom_gate_calibration_loader is not None:
            log_feature_cache_loader(atom_gate_calibration_loader, "atom_gate_calibration_loader")
        if score_bias_calibration_loader is not None:
            log_feature_cache_loader(score_bias_calibration_loader, "score_bias_calibration_loader")

        if len(self._multiple_gpus) > 1 and self._network is not None:
            logging.info("Using multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(
            self.train_loader,
            self.test_loader,
            self.train_loader_for_hc,
            fallback_calibration_loader=fallback_calibration_loader,
            atom_gate_calibration_loader=atom_gate_calibration_loader,
            score_bias_calibration_loader=score_bias_calibration_loader,
        )

        if len(self._multiple_gpus) > 1 and self._network is not None:
            self._network = self._network.module

    def _train(
        self,
        train_loader,
        test_loader,
        train_loader_for_hc,
        fallback_calibration_loader=None,
        atom_gate_calibration_loader=None,
        score_bias_calibration_loader=None,
    ):
        if self._network is not None:
            self._network.to(self._device)
        class_feature_data = self._extract_class_features(train_loader_for_hc, self._network)
        self._compress_task_boundary()
        self._fit_class_score_normalization(class_feature_data)
        self._calibrate_score_bias(score_bias_calibration_loader)
        self._calibrate_atom_conflict_gate(atom_gate_calibration_loader)
        self._calibrate_raw_fallback_gate(fallback_calibration_loader)

    def _fit_class_score_normalization(self, class_feature_data):
        if not bool(getattr(self.hc_soinn, "use_class_score_normalization", False)):
            return
        if class_feature_data is None:
            logging.warning("[LifeTopoDict] Class score normalization skipped: no train features.")
            return
        feats, labels = class_feature_data
        if hasattr(self.hc_soinn, "fit_class_score_normalization"):
            self.hc_soinn.fit_class_score_normalization(feats, labels)

    def _calibrate_score_bias(self, calibration_loader):
        if calibration_loader is None:
            return
        if not bool(getattr(self.hc_soinn, "use_score_bias_calibration", False)):
            return
        boundary = getattr(self.hc_soinn, "known_classes_before_task", None)
        if boundary is None or int(boundary) <= 0:
            self.hc_soinn.score_bias_new = 0.0
            self.hc_soinn._score_bias_fit_stats = {
                "samples": 0.0,
                "best_bias": 0.0,
                "calibration_accuracy": 0.0,
                "calibration_compact_accuracy": 0.0,
                "calibration_gain": 0.0,
                "gate_disabled": 1.0,
            }
            return

        grid = tuple(getattr(self.hc_soinn, "score_bias_grid", (0.0,)))
        if not grid:
            grid = (0.0,)
        logging.info(
            "[LifeTopoDict] Calibrating compact score bias on %d samples (grid=%s).",
            len(calibration_loader.dataset),
            ",".join(f"{float(v):.4f}" for v in grid),
        )

        original_bias = float(getattr(self.hc_soinn, "score_bias_new", 0.0))
        original_trace = bool(getattr(self.hc_soinn, "enable_prediction_trace", False))
        self.hc_soinn.enable_prediction_trace = False

        scores = []
        try:
            for bias in grid:
                self.hc_soinn.score_bias_new = float(bias)
                y_pred, y_true = self._eval_cnn(calibration_loader)
                if y_pred.shape[0] == 0:
                    acc = 0.0
                else:
                    acc = float(np.mean(y_pred[:, 0] == y_true))
                scores.append((float(bias), acc))
        finally:
            self.hc_soinn.enable_prediction_trace = original_trace

        zero_acc = None
        for bias, acc in scores:
            if abs(bias) <= 1e-12:
                zero_acc = acc
                break
        if zero_acc is None:
            self.hc_soinn.score_bias_new = 0.0
            y_pred, y_true = self._eval_cnn(calibration_loader)
            zero_acc = float(np.mean(y_pred[:, 0] == y_true)) if y_pred.shape[0] > 0 else 0.0
            scores.append((0.0, zero_acc))

        best_bias, best_acc = 0.0, zero_acc
        for bias, acc in scores:
            if acc > best_acc + 1e-12 or (
                abs(acc - best_acc) <= 1e-12 and abs(bias) < abs(best_bias)
            ):
                best_bias, best_acc = float(bias), float(acc)

        gain = float(best_acc - zero_acc)
        disabled = 0.0
        if gain < float(getattr(self.hc_soinn, "score_bias_min_gain", 0.0)):
            best_bias = 0.0
            best_acc = zero_acc
            gain = 0.0
            disabled = 1.0

        self.hc_soinn.score_bias_new = float(best_bias)
        self.hc_soinn._score_bias_fit_stats = {
            "samples": float(len(calibration_loader.dataset)),
            "best_bias": float(best_bias),
            "calibration_accuracy": float(best_acc),
            "calibration_compact_accuracy": float(zero_acc),
            "calibration_gain": float(gain),
            "gate_disabled": float(disabled),
            "original_bias": float(original_bias),
        }
        logging.info(
            "[LifeTopoDict] Score bias fit: samples=%d, compact_acc=%.4f, "
            "calib_acc=%.4f, gain=%.4f, bias=%.5f, disabled=%d",
            int(self.hc_soinn._score_bias_fit_stats["samples"]),
            float(zero_acc),
            float(best_acc),
            float(gain),
            float(best_bias),
            int(disabled),
        )

    def _calibrate_atom_conflict_gate(self, calibration_loader):
        if calibration_loader is None:
            return
        if not hasattr(self.hc_soinn, "fit_atom_conflict_gate_from_trace"):
            return
        if not bool(getattr(self.hc_soinn, "use_atom_conflict_gate", False)):
            return
        logging.info(
            "[LifeTopoDict] Calibrating atom conflict gate on %d samples "
            "(metric=%s, target=%s, trace_split=%s).",
            len(calibration_loader.dataset),
            str(getattr(self.hc_soinn, "atom_conflict_metric", "pmi")),
            str(getattr(self.hc_soinn, "atom_conflict_target", "all_errors")),
            getattr(self.hc_soinn, "trace_split", "test"),
        )
        _ = self._eval_cnn(calibration_loader)
        trace_records = self.hc_soinn.get_prediction_trace()
        self._dump_prediction_trace_records(trace_records, split="atom_calibration")
        fit_stats = self.hc_soinn.fit_atom_conflict_gate_from_trace(trace_records)
        if fit_stats:
            logging.info(
                "[LifeTopoDict] Atom conflict gate fit: samples=%d, errors=%.0f, "
                "global_error=%.4f, pairs=%d, deployed=%d, metric=%s, target=%s, "
                "strength=%.4f, calib_acc=%.4f, compact_acc=%.4f, gain=%.4f, "
                "gate_rate=%.4f, benefit_sel=%.0f, harm_sel=%.0f, disabled=%d",
                int(fit_stats.get("samples", 0.0)),
                float(fit_stats.get("errors", 0.0)),
                float(fit_stats.get("global_error_rate", 0.0)),
                int(fit_stats.get("pair_count", 0.0)),
                int(fit_stats.get("deployed_pairs", 0.0)),
                str(fit_stats.get("metric", "")),
                str(fit_stats.get("target", "")),
                float(fit_stats.get("strength", 0.0)),
                float(fit_stats.get("calibration_accuracy", 0.0)),
                float(fit_stats.get("compact_accuracy", 0.0)),
                float(fit_stats.get("calibration_gain", 0.0)),
                float(fit_stats.get("calibration_gate_rate", 0.0)),
                float(fit_stats.get("calibration_benefit_selected", 0.0)),
                float(fit_stats.get("calibration_harm_selected", 0.0)),
                int(fit_stats.get("gate_disabled", 0.0)),
            )

    def _calibrate_raw_fallback_gate(self, calibration_loader):
        if calibration_loader is None:
            return
        if not hasattr(self.hc_soinn, "fit_raw_fallback_gate_from_trace"):
            return
        gate_type = str(getattr(self.hc_soinn, "fallback_gate_type", "rule")).lower()
        if gate_type not in (
            "ridge", "logistic", "gbdt", "gbdt_depth2", "random_forest", "rf", "rf_depth4",
            "candidate_rule", "delta_ridge", "delta_logistic", "benefit_harm_logistic",
            "dual_logistic", "dual_logistic_lcb", "utility_tree", "utility_gbdt",
            "utility_table", "delta_ridge_pair",
        ):
            return
        logging.info(
            "[LifeTopoDict] Calibrating raw fallback gate on %d samples (type=%s, trace_split=%s).",
            len(calibration_loader.dataset),
            gate_type,
            getattr(self.hc_soinn, "trace_split", "test"),
        )
        _ = self._eval_cnn(calibration_loader)
        trace_records = self.hc_soinn.get_prediction_trace()
        self._dump_prediction_trace_records(trace_records, split="calibration")
        fit_stats = self.hc_soinn.fit_raw_fallback_gate_from_trace(trace_records)
        if fit_stats:
            logging.info(
                "[LifeTopoDict] Raw fallback gate fit: samples=%d, positives=%.0f, "
                "calib_acc=%.4f, compact_acc=%.4f, fallback_acc=%.4f, oracle_acc=%.4f, "
                "gain=%.4f, net=%.4f, fallback_rate=%.4f, benefit_sel=%.0f, harm_sel=%.0f, "
                "precision=%.4f, threshold=%.6f, weight_l2=%.4f, disabled=%d",
                int(fit_stats.get("samples", 0.0)),
                float(fit_stats.get("positives", 0.0)),
                float(fit_stats.get("calibration_accuracy", 0.0)),
                float(fit_stats.get("compact_accuracy", 0.0)),
                float(fit_stats.get("fallback_accuracy", 0.0)),
                float(fit_stats.get("oracle_accuracy", 0.0)),
                float(fit_stats.get("calibration_gain", 0.0)),
                float(fit_stats.get("calibration_net_gain", 0.0)),
                float(fit_stats.get("calibration_fallback_rate", 0.0)),
                float(fit_stats.get("calibration_benefit_selected", 0.0)),
                float(fit_stats.get("calibration_harm_selected", 0.0)),
                float(fit_stats.get("calibration_utility_precision", 0.0)),
                float(fit_stats.get("threshold", 0.0)),
                float(fit_stats.get("weight_l2", 0.0)),
                int(fit_stats.get("gate_disabled", 0.0)),
            )

    # ------------------------------------------------------------------ #
    # Evaluation / inference                                              #
    # ------------------------------------------------------------------ #
    def _eval_cnn(self, loader):
        y_pred, y_true = [], []
        if hasattr(self.hc_soinn, "reset_raw_fallback_eval_stats"):
            self.hc_soinn.reset_raw_fallback_eval_stats(clear_trace=True)
        if hasattr(self.hc_soinn, "reset_raw_auxiliary_eval_stats"):
            self.hc_soinn.reset_raw_auxiliary_eval_stats()
        if hasattr(self.hc_soinn, "reset_atom_conflict_eval_stats"):
            self.hc_soinn.reset_atom_conflict_eval_stats()
        from_cache = is_cached_feature_loader(loader)
        if not from_cache and self._network is None:
            raise RuntimeError(
                "Backbone is skipped, but test_loader is not backed by cached features."
            )
        if self._network is not None:
            self._network.eval()

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                if from_cache:
                    features = tensor2numpy(inputs.float())
                else:
                    inputs = inputs.to(self._device)
                    features = self._network.extract_vector(inputs)
                    features = tensor2numpy(features)

                topk_pred = self.hc_soinn.predict_topk(
                    features,
                    self.topk,
                    self._total_classes,
                    device=self._classifier_device,
                    targets=targets.numpy(),
                )
                y_pred.append(topk_pred)
                y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        self._dump_prediction_trace(split="test")
        acc = self._evaluate(y_pred, y_true)
        return {"hc_soinn": acc}

    def _json_safe(self, value):
        if isinstance(value, np.generic):
            return self._json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [self._json_safe(v) for v in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, float):
            if not np.isfinite(value):
                return None
            return value
        return value

    def _dump_prediction_trace(self, split="test"):
        output_dir = self._prediction_trace_output_dir
        if not output_dir:
            return
        if not bool(getattr(self.hc_soinn, "enable_prediction_trace", False)):
            return
        if not self._prediction_trace_dump_all_tasks:
            # Without a reliable final-task callback here, the default keeps all
            # task traces. This branch is reserved for future explicit callers.
            return

        records = self.hc_soinn.get_prediction_trace()
        if not records:
            logging.info("[PredictionTrace] No records to dump for task=%s", self._cur_task)
            return

        trace_split = str(getattr(self.hc_soinn, "trace_split", split))
        self._dump_prediction_trace_records(records, split=trace_split)

    def _dump_prediction_trace_records(self, records, split="test"):
        output_dir = self._prediction_trace_output_dir
        if not output_dir:
            return
        if not bool(getattr(self.hc_soinn, "enable_prediction_trace", False)):
            return
        if not records:
            return

        os.makedirs(output_dir, exist_ok=True)
        prefix = str(self.args.get("prefix", "life_topo_dict"))
        dataset = str(self.args.get("dataset", "dataset"))
        seed = str(self.args.get("seed", "seed"))
        trace_split = str(split)
        filename = (
            f"{dataset}_{prefix}_seed{seed}_task{self._cur_task:02d}_"
            f"classes{self._total_classes}_{trace_split}.jsonl"
        )
        path = os.path.join(output_dir, filename)
        metadata = {
            "dataset": dataset,
            "prefix": prefix,
            "seed": self.args.get("seed", None),
            "task": int(self._cur_task),
            "known_classes": int(self._known_classes),
            "total_classes": int(self._total_classes),
            "init_cls": int(self.args.get("init_cls", 0)),
            "increment": int(self.args.get("increment", 0)),
            "trace_split": trace_split,
            "fallback_gate_type": str(getattr(self.hc_soinn, "fallback_gate_type", "")),
            "raw_fallback_per_class": int(getattr(self.hc_soinn, "raw_fallback_per_class", 0)),
            "raw_fallback_select_by": str(getattr(self.hc_soinn, "raw_fallback_select_by", "")),
        }
        with open(path, "w", encoding="utf-8") as f:
            for idx, record in enumerate(records):
                out = dict(metadata)
                out["sample_index"] = int(idx)
                out.update(record)
                f.write(json.dumps(self._json_safe(out), ensure_ascii=False) + "\n")
        logging.info(
            "[PredictionTrace] Dumped %d records to %s",
            len(records),
            path,
        )
