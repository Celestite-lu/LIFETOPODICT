import json
import argparse
from trainer import train


def _str2bool(v):
    """Parse boolean CLI argument: accept true/false/yes/no/1/0."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}')


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args_dict = vars(args) # Converting argparse Namespace to a dict.
    
    # Save command line device if specified
    cmd_device = None
    if args_dict.get('device') is not None:
        # Convert device strings to integers
        cmd_device = [int(d) for d in args_dict['device']]

    # Save command line optional overrides (non-None values only)
    _ltd_overrides = {}
    for _key in [
        'dict_sparse_k', 'dict_ridge_lambda',
        'drop_node_raw_after_dict_materialize',
        'lifecycle_theta_support', 'lifecycle_theta_usage', 'lifecycle_T_inactive',
        'lifecycle_min_alive_ratio', 'lifecycle_protect_old_topk',
        'lifecycle_node_inactive_threshold',
        'dict_theta_residual', 'dict_max_growth_per_task',
        'use_dict_coding', 'use_dictionary_growth', 'use_lifecycle', 'use_protected_gate',
        'use_edge_age_persistence',
        'use_edge_aware_scoring', 'edge_score_gamma', 'edge_score_eta',
        'use_raw_fallback_gate', 'raw_fallback_per_class',
        'raw_fallback_select_by', 'fallback_gate_type',
        'fallback_gate_margin_threshold', 'fallback_gate_residual_threshold',
        'fallback_gate_random_rate', 'fallback_memory_accounting',
        'fallback_calibration_samples_per_class',
        'fallback_calibration_cumulative',
        'fallback_gate_min_positives', 'fallback_gate_min_calibration_gain',
        'fallback_gate_max_rate', 'fallback_gate_tree_estimators',
        'fallback_gate_tree_max_depth', 'fallback_gate_tree_min_samples_leaf',
        'fallback_gate_threshold_strategy', 'fallback_gate_min_precision',
        'fallback_gate_min_selected', 'fallback_gate_min_net_count',
        'fallback_candidate_rule', 'fallback_gate_budget_rates',
        'fallback_gate_neutral_weight', 'fallback_pair_table_mode',
        'fallback_pair_table_smoothing', 'fallback_pair_table_lambda',
        'fallback_node_table_lambda',
        'use_raw_auxiliary_nodes', 'raw_auxiliary_penalty',
        'enable_prediction_trace', 'trace_split', 'fallback_random_seed',
        'prediction_trace_output_dir', 'prediction_trace_dump_all_tasks',
        'use_atom_conflict_gate', 'atom_conflict_metric',
        'atom_conflict_target', 'atom_conflict_topk',
        'atom_gate_strength', 'atom_gate_min_pair_support',
        'atom_gate_smoothing', 'atom_gate_calibration_samples_per_class',
        'atom_gate_calibration_cumulative',
        'use_score_bias_calibration', 'score_bias_calibration_samples_per_class',
        'score_bias_calibration_cumulative', 'score_bias_grid',
        'score_bias_min_gain',
        'use_node_residual_penalty', 'node_residual_penalty_strength',
        'node_residual_penalty_mode',
        'use_feature_cache', 'feature_cache_dir', 'feature_cache_strict',
        'feature_cache_dtype', 'feature_cache_skip_backbone',
        'feature_cache_classifier_device',
        'visualize_tsne',
        'training_time_profile_last_task', 'skip_eval_for_training_time_profile',
        'seed',
        'max_tasks',
    ]:
        if args_dict.get(_key) is not None:
            _ltd_overrides[_key] = args_dict[_key]

    # Merge JSON config (JSON values take precedence for most params)
    args_dict.update(param)

    # Override with command line device if specified
    if cmd_device is not None:
        args_dict['device'] = cmd_device

    # Override with command line optional params if specified
    for _key, _val in _ltd_overrides.items():
        args_dict[_key] = _val

    # Drop unset CLI defaults so downstream args.get(..., default) fallbacks still apply.
    args_dict = {key: value for key, value in args_dict.items() if value is not None}
    
    # Ensure device is in correct format (list of ints)
    if isinstance(args_dict.get('device'), list) and len(args_dict.get('device', [])) > 0:
        if isinstance(args_dict['device'][0], str):
            args_dict['device'] = [int(d) for d in args_dict['device']]

    train(args_dict)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--device', type=str, nargs='+', default=None,
                        help='Device(s) to use. Can specify multiple devices, e.g., --device 0 or --device 0 1. Use -1 for CPU.')
    parser.add_argument('--seed', type=int, nargs='+', default=None,
                        help='Override random seed list from JSON config, e.g., --seed 1993 or --seed 1993 1994.')
    parser.add_argument('--max_tasks', type=int, default=None,
                        help='Limit number of incremental tasks for smoke runs. Default uses all tasks.')

    # --- LifeTopoDict parameters (can override JSON config via CLI) ---
    parser.add_argument('--dict_sparse_k', type=int, default=None,
                        help='Top-K sparse coding support size (default: 5).')
    parser.add_argument('--dict_ridge_lambda', type=float, default=None,
                        help='Ridge regression lambda for sparse coding (default: 0.1).')
    parser.add_argument('--drop_node_raw_after_dict_materialize', type=_str2bool, default=None,
                        help='Drop redundant node raw/reconstruction vectors after dictionary materialization.')
    parser.add_argument('--lifecycle_theta_support', type=float, default=None,
                        help='Old-class support threshold for protected state (default: 0.5).')
    parser.add_argument('--lifecycle_theta_usage', type=float, default=None,
                        help='Usage rate threshold for inactive detection (default: 0.01).')
    parser.add_argument('--lifecycle_T_inactive', type=int, default=None,
                        help='Consecutive low-usage tasks before marking inactive (default: 3).')
    parser.add_argument('--lifecycle_min_alive_ratio', type=float, default=None,
                        help='Minimum non-inactive atom ratio after lifecycle safety (default: 0.70).')
    parser.add_argument('--lifecycle_protect_old_topk', type=_str2bool, default=None,
                        help='Protect atoms used by old-class top-k sparse codes (default: true).')
    parser.add_argument('--lifecycle_node_inactive_threshold', type=float, default=None,
                        help='Inactive top-k atom fraction needed to mark a node inactive (default: 1.0).')
    parser.add_argument('--dict_theta_residual', type=float, default=None,
                        help='Residual threshold triggering dictionary growth (default: 0.3).')
    parser.add_argument('--dict_max_growth_per_task', type=int, default=None,
                        help='Maximum new atoms added per task. Deprecated for forward experiments; default is 0.')

    # --- P0-4: Ablation switches (default None = use JSON config or True) ---
    parser.add_argument('--use_dict_coding', type=_str2bool, default=None,
                        help='Enable shared dictionary sparse coding (default: true for LifeTopoDict).')
    parser.add_argument('--use_dictionary_growth', type=_str2bool, default=None,
                        help='Enable residual-triggered dictionary growth. Deprecated for forward experiments; default is false.')
    parser.add_argument('--use_lifecycle', type=_str2bool, default=None,
                        help='Enable lifecycle state transitions (default: true).')
    parser.add_argument('--use_protected_gate', type=_str2bool, default=None,
                        help='Enable protected node deletion gate (default: true).')
    parser.add_argument('--use_edge_age_persistence', type=_str2bool, default=None,
                        help='Persist SOINN edge ages and reliability metadata (default: true).')
    parser.add_argument('--use_edge_aware_scoring', type=_str2bool, default=None,
                        help='Enable additive edge-aware local score adjustment. Deprecated for forward experiments; default is false.')
    parser.add_argument('--edge_score_gamma', type=float, default=None,
                        help='Edge support weight for edge-aware scoring (default: 0.1).')
    parser.add_argument('--edge_score_eta', type=float, default=None,
                        help='Edge age/risk penalty for edge-aware scoring (default: 0.05).')

    # --- Direction 1: reliability-calibrated raw-node fallback gate ---
    parser.add_argument('--use_raw_fallback_gate', type=_str2bool, default=None,
                        help='Enable audited raw-node fallback gate for LifeTopoDict.')
    parser.add_argument('--raw_fallback_per_class', type=int, default=None,
                        help='Maximum raw fallback node snapshots kept per class.')
    parser.add_argument('--raw_fallback_select_by', type=str, default=None,
                        help='Fallback node ranking: residual, residual_margin, risk, or random.')
    parser.add_argument('--fallback_gate_type', type=str, default=None,
                        help='Gate type: rule, random, static, oracle, logistic, ridge, gbdt, or random_forest.')
    parser.add_argument('--fallback_gate_margin_threshold', type=float, default=None,
                        help='Rule gate margin threshold; lower compact margin triggers fallback.')
    parser.add_argument('--fallback_gate_residual_threshold', type=float, default=None,
                        help='Rule gate residual threshold; higher selected-node residual triggers fallback.')
    parser.add_argument('--fallback_gate_random_rate', type=float, default=None,
                        help='Random gate fallback probability for random-control runs.')
    parser.add_argument('--fallback_memory_accounting', type=_str2bool, default=None,
                        help='Include raw fallback cache in compact deployable memory accounting.')
    parser.add_argument('--fallback_calibration_samples_per_class', type=int, default=None,
                        help='Per-class holdout samples used to fit learned fallback gates.')
    parser.add_argument('--fallback_calibration_cumulative', type=_str2bool, default=None,
                        help='Fit learned fallback gates on cumulative held-out calibration samples.')
    parser.add_argument('--fallback_gate_min_positives', type=int, default=None,
                        help='Disable learned fallback gate when calibration positives are below this count.')
    parser.add_argument('--fallback_gate_min_calibration_gain', type=float, default=None,
                        help='Disable learned fallback gate unless calibration gain over compact exceeds this value.')
    parser.add_argument('--fallback_gate_max_rate', type=float, default=None,
                        help='Maximum calibration fallback rate allowed when selecting learned gate threshold.')
    parser.add_argument('--fallback_gate_tree_estimators', type=int, default=None,
                        help='Number of estimators for tree-based fallback gates.')
    parser.add_argument('--fallback_gate_tree_max_depth', type=int, default=None,
                        help='Maximum tree depth for tree-based fallback gates.')
    parser.add_argument('--fallback_gate_tree_min_samples_leaf', type=int, default=None,
                        help='Minimum leaf samples for tree-based fallback gates.')
    parser.add_argument('--fallback_gate_threshold_strategy', type=str, default=None,
                        help='Threshold selection strategy for utility gates: max_net or lcb.')
    parser.add_argument('--fallback_gate_min_precision', type=float, default=None,
                        help='Minimum calibration utility precision for selected fallback rows.')
    parser.add_argument('--fallback_gate_min_selected', type=int, default=None,
                        help='Minimum calibration selected fallback rows for a learned gate.')
    parser.add_argument('--fallback_gate_min_net_count', type=int, default=None,
                        help='Minimum calibration benefit-minus-harm count for a learned gate.')
    parser.add_argument('--fallback_candidate_rule', type=str, default=None,
                        help='Deployable candidate filter before learned fallback gating: none, disagree, margin_raw, enhanced.')
    parser.add_argument('--fallback_gate_budget_rates', type=str, default=None,
                        help='Comma-separated validation fallback budgets, e.g. 0.002,0.005,0.01,0.02,0.04.')
    parser.add_argument('--fallback_gate_neutral_weight', type=float, default=None,
                        help='Sample weight for neutral delta=0 rows in utility gates.')
    parser.add_argument('--fallback_pair_table_mode', type=str, default=None,
                        help='Utility shrinkage table mode: none, class_pair, class_pair_task, node_pair, full.')
    parser.add_argument('--fallback_pair_table_smoothing', type=float, default=None,
                        help='Smoothing denominator for pair/node utility tables.')
    parser.add_argument('--fallback_pair_table_lambda', type=float, default=None,
                        help='Score weight for class-pair/task utility table backoff.')
    parser.add_argument('--fallback_node_table_lambda', type=float, default=None,
                        help='Score weight for selected-node/raw-class utility table.')
    parser.add_argument('--use_raw_auxiliary_nodes', type=_str2bool, default=None,
                        help='Use audited raw node snapshots as penalized auxiliary local HC-SOINN candidates.')
    parser.add_argument('--raw_auxiliary_penalty', type=float, default=None,
                        help='Additive cosine-distance penalty applied to raw auxiliary nodes.')
    parser.add_argument('--enable_prediction_trace', type=_str2bool, default=None,
                        help='Collect compact/fallback prediction trace summaries during evaluation.')
    parser.add_argument('--trace_split', type=str, default=None,
                        help='Trace split label, e.g. val or test.')
    parser.add_argument('--fallback_random_seed', type=int, default=None,
                        help='Random seed for random fallback gate/control selection.')
    parser.add_argument('--prediction_trace_output_dir', type=str, default=None,
                        help='Directory for JSONL prediction trace dumps.')
    parser.add_argument('--prediction_trace_dump_all_tasks', type=_str2bool, default=None,
                        help='Dump prediction trace after every eval task when trace output is enabled.')

    parser.add_argument('--use_score_bias_calibration', type=_str2bool, default=None,
                        help='Fit a calibration split scalar old/new score bias for compact HC-SOINN scores.')
    parser.add_argument('--score_bias_calibration_samples_per_class', type=int, default=None,
                        help='Per-class held-out train samples used to fit the score bias scalar.')
    parser.add_argument('--score_bias_calibration_cumulative', type=_str2bool, default=None,
                        help='Fit score bias on cumulative held-out calibration samples.')
    parser.add_argument('--score_bias_grid', type=str, default=None,
                        help='Comma-separated candidate new-class score biases; lower scores are better.')
    parser.add_argument('--score_bias_min_gain', type=float, default=None,
                        help='Minimum calibration accuracy gain required to deploy a nonzero score bias.')
    parser.add_argument('--use_node_residual_penalty', type=_str2bool, default=None,
                        help='Penalize compact topology node distances by dictionary reconstruction residual.')
    parser.add_argument('--node_residual_penalty_strength', type=float, default=None,
                        help='Strength for node residual distance penalty.')
    parser.add_argument('--node_residual_penalty_mode', type=str, default=None,
                        help='Node residual penalty mode: linear, sqrt, or random_control.')

    # --- Direction 2: ATD-aware atom conflict gate ---
    parser.add_argument('--use_atom_conflict_gate', type=_str2bool, default=None,
                        help='Enable calibration-fit atom/class conflict reliability gate.')
    parser.add_argument('--atom_conflict_metric', type=str, default=None,
                        help='Atom conflict ranking: pmi, old_new_pmi, random, high_usage, or old_support.')
    parser.add_argument('--atom_conflict_target', type=str, default=None,
                        help='Calibration target for atom conflict: all_errors or old_new.')
    parser.add_argument('--atom_conflict_topk', type=int, default=None,
                        help='Number of positive atom/class conflict entries to deploy.')
    parser.add_argument('--atom_gate_strength', type=float, default=None,
                        help='Distance penalty multiplier for atom conflict score.')
    parser.add_argument('--atom_gate_min_pair_support', type=int, default=None,
                        help='Minimum calibration activations for an atom/class pair.')
    parser.add_argument('--atom_gate_smoothing', type=float, default=None,
                        help='Empirical-Bayes smoothing denominator for atom conflict rates.')
    parser.add_argument('--atom_gate_calibration_samples_per_class', type=int, default=None,
                        help='Per-class held-out train samples used to fit atom conflict gate.')
    parser.add_argument('--atom_gate_calibration_cumulative', type=_str2bool, default=None,
                        help='Fit atom conflict gate on cumulative held-out calibration samples.')

    # --- Feature cache parameters (can override JSON config via CLI) ---
    parser.add_argument('--use_feature_cache', type=_str2bool, default=None,
                        help='Use offline frozen-backbone feature cache when available.')
    parser.add_argument('--feature_cache_dir', type=str, default=None,
                        help='Feature cache root directory.')
    parser.add_argument('--feature_cache_strict', type=_str2bool, default=None,
                        help='Raise an error when feature cache is enabled but missing/incompatible.')
    parser.add_argument('--feature_cache_dtype', type=str, choices=['float32', 'float16'], default=None,
                        help='Optional feature cache dtype record/override.')
    parser.add_argument('--feature_cache_skip_backbone', type=_str2bool, default=None,
                        help='Skip backbone initialization in strict feature-cache classifier-only runs.')
    parser.add_argument('--feature_cache_classifier_device', type=str, default=None,
                        help='Device for cached-feature classifier inference: cpu, cuda, or explicit torch device.')

    # --- Optional diagnostics ---
    parser.add_argument('--visualize_tsne', type=_str2bool, default=None,
                        help='Enable HC-SOINN t-SNE visualization diagnostics (default: false).')
    parser.add_argument('--training_time_profile_last_task', type=_str2bool, default=None,
                        help='Enable training-only profiling path supported by trainer.py.')
    parser.add_argument('--skip_eval_for_training_time_profile', type=_str2bool, default=None,
                        help='Skip evaluation while training_time_profile_last_task is enabled.')

    return parser

if __name__ == '__main__':
    main()
