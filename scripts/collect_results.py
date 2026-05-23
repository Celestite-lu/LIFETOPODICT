#!/usr/bin/env python3
"""Collect experiment results from log files and output as CSV (Agent I).

Reads one or more LifeTopoDict (or HC-SOINN) experiment log files produced
by ``trainer.py`` and extracts:

- Per-task accuracy curves (HC-SOINN top1 / top5)
- Final accuracy
- Average Accuracy
- Forgetting
- LifeTopoDict diagnostic metrics (PAD, EffRank, GTE, atom_count, etc.)

Usage:
    python scripts/collect_results.py LOG_DIR [LOG_DIR ...] -o results.csv
    python scripts/collect_results.py path/to/specific.log -o results.csv
    python scripts/collect_results.py logs/life_topo_dict/ -o results.csv --recursive
"""

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional


# ====================================================================== #
# Log parsing helpers                                                     #
# ====================================================================== #

# Patterns for extracting metrics from log lines
_RE_ACC_TOP1 = re.compile(
    r'HC-SOINN top1 curve: \[(.*?)\]'
)
_RE_ACC_TOP5 = re.compile(
    r'HC-SOINN top5 curve: \[(.*?)\]'
)
_RE_AVG_ACC = re.compile(
    r'Average Accuracy \(HC-SOINN\): ([-\d.eE+]+)'
)
_RE_FORGETTING = re.compile(
    r'Forgetting \(HC-SOINN\): ([-\d.eE+]+)'
)
_RE_DIAG_PAD = re.compile(
    r'PAD=([-\d.eEnaN+]+)'
)
_RE_DIAG_EFFRANK = re.compile(
    r'EffRank=([-\d.eEnaN+]+)'
)
_RE_DIAG_GTE = re.compile(
    r'GTE=([-\d.eEnaN+]+)'
)
_RE_DIAG_ATOMS = re.compile(
    r'atoms=(\d+)'
)
_RE_DIAG_PROTECTED = re.compile(
    r'protected=([\d.]+)%'
)
_RE_DIAG_INACTIVE = re.compile(
    r'inactive=([\d.]+)%'
)
_RE_DIAG_INACTIVE_NODES = re.compile(
    r'inactive_nodes=([\d.]+)%'
)
_RE_DIAG_FILTERED_INACTIVE_NODES = re.compile(
    r'filtered_inactive_nodes=([\d.]+)%'
)
_RE_DIAG_RESIDUAL = re.compile(
    r'avg_residual=([-\d.eEnaN+]+)'
)
_RE_EDGE_LINE = re.compile(
    r'\[LifeTopoDict\] Edge scoring:'
)
_RE_EDGE_USE_RATE = re.compile(
    r'use_rate=([-\d.eEnaN+]+)'
)
_RE_EDGE_CLASS_USE_RATE = re.compile(
    r'class_use_rate=([-\d.eEnaN+]+)'
)
_RE_EDGE_MARGIN = re.compile(
    r'margin_contribution=([-\d.eEnaN+]+)'
)
_RE_EDGE_RISK = re.compile(
    r'risk_penalty=([-\d.eEnaN+]+)'
)
_RE_EDGE_ADJUSTMENT = re.compile(
    r'avg_adjustment=([-\d.eEnaN+]+)'
)
_RE_RAW_FALLBACK_LINE = re.compile(
    r'\[LifeTopoDict\] Raw fallback:'
)
_RE_RAW_FALLBACK_VALUE = re.compile(
    r'([A-Za-z_]+)=([-\d.eEnaN+]+)'
)
_RE_RAW_AUXILIARY_LINE = re.compile(
    r'\[LifeTopoDict\] Raw auxiliary:'
)
_RE_RAW_AUXILIARY_VALUE = re.compile(
    r'([A-Za-z_]+)=([-\d.eEnaN+]+)'
)
_RE_RAW_FALLBACK_GATE_FIT_LINE = re.compile(
    r'\[LifeTopoDict\] Raw fallback gate fit:'
)
_RE_RAW_FALLBACK_GATE_VALUE = re.compile(
    r'([A-Za-z_]+)=([-\d.eEnaN+]+)'
)
_RE_ATOM_CONFLICT_GATE_FIT_LINE = re.compile(
    r'\[LifeTopoDict\] Atom conflict gate fit:'
)
_RE_ATOM_CONFLICT_GATE_VALUE = re.compile(
    r'([A-Za-z_]+)=([-\d.eEnaN+]+)'
)
_RE_SCORE_BIAS_FIT_LINE = re.compile(
    r'\[LifeTopoDict\] Score bias fit:'
)
_RE_SCORE_BIAS_VALUE = re.compile(
    r'([A-Za-z_]+)=([-\d.eEnaN+]+)'
)
_RE_PAIR_MARGIN_FIT_LINE = re.compile(
    r'\[LifeTopoDict\] Pair margin fit:'
)
_RE_PAIR_MARGIN_VALUE = re.compile(
    r'([A-Za-z_]+)=([-\d.eEnaN+]+)'
)
_RE_TOPOLOGY_RELIABILITY_LINE = re.compile(
    r'\[LifeTopoDict\] Topology reliability arbitration:'
)
_RE_TOPOLOGY_RELIABILITY_VALUE = re.compile(
    r'([A-Za-z_]+)=([-\d.eEnaN+]+)'
)
_RE_CLASS_RESIDUAL_REPAIR_LINE = re.compile(
    r'\[LifeTopoDict\] Class residual repair:'
)
_RE_CLASS_RESIDUAL_REPAIR_VALUE = re.compile(
    r'([A-Za-z_]+)=([-\d.eEnaN+]+)'
)
_RE_CONFIG_PREFIX = re.compile(
    r'prefix:[ \t]*(.*)'
)
_RE_DATASET = re.compile(
    r'dataset: (\S+)'
)
_RE_SEED = re.compile(
    r'seed: \[?(\d+)\]?'
)
_RE_BACKBONE = re.compile(
    r'backbone_type: (\S+)'
)
_RE_INIT_CLS = re.compile(
    r'init_cls: (\d+)'
)
_RE_INCREMENT = re.compile(
    r'increment: (\d+)'
)
_RE_MEMORY_LINE = re.compile(
    r'\[(?:LifeTopoDict|HC-SOINN)\] Memory:'
)
_RE_MEMORY_VALUE = re.compile(
    r'([A-Za-z_]+)=([-\d.eE+]+) MB'
)
_RE_FLOAT_TOKEN = re.compile(
    r'np\.float64\(([-\d.eE+]+)\)|(?<![A-Za-z0-9_.])([-\d]+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
)

# Simpler diagnostic line: the full diagnostics block
_RE_DIAG_LINE = re.compile(
    r'\[LifeTopoDict\] Diagnostics:'
)


def _safe_float(s: str) -> Optional[float]:
    """Parse a float string that may contain 'nan' or similar."""
    s = s.strip()
    if s in ('nan', 'NaN', 'NA', ''):
        return float('nan')
    try:
        return float(s)
    except ValueError:
        return None


def _parse_float_list(s: str) -> List[float]:
    values: List[float] = []
    for match in _RE_FLOAT_TOKEN.findall(s):
        token = match[0] or match[1]
        parsed = _safe_float(token)
        if parsed is not None:
            values.append(parsed)
    return values


def _last_match(pattern: re.Pattern, content: str):
    matches = list(pattern.finditer(content))
    return matches[-1] if matches else None


def _seed_blocks(content: str) -> List[str]:
    starts = [m.start() for m in _RE_SEED.finditer(content)]
    if not starts:
        return [content]
    starts.append(len(content))
    return [content[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def _first_group(pattern: re.Pattern, content: str, cast=None):
    m = pattern.search(content)
    if not m:
        return None
    value = m.group(1).strip() if isinstance(m.group(1), str) else m.group(1)
    return cast(value) if cast is not None else value


def _global_metadata(content: str) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    for key, pattern, cast in [
        ('prefix', _RE_CONFIG_PREFIX, str),
        ('dataset', _RE_DATASET, str),
        ('backbone', _RE_BACKBONE, str),
        ('init_cls', _RE_INIT_CLS, int),
        ('increment', _RE_INCREMENT, int),
    ]:
        value = _first_group(pattern, content, cast)
        if value is not None:
            meta[key] = value
    return meta


def _parse_log_block(log_path: str, content: str, block_index: int) -> Dict[str, object]:
    result: Dict[str, object] = {
        'log_file': os.path.basename(log_path),
    }
    if block_index > 0:
        result['seed_block'] = block_index

    # --- Config params ---
    m = _last_match(_RE_CONFIG_PREFIX, content)
    if m:
        result['prefix'] = m.group(1)

    m = _last_match(_RE_DATASET, content)
    if m:
        result['dataset'] = m.group(1)

    m = _last_match(_RE_SEED, content)
    if m:
        result['seed'] = int(m.group(1))

    m = _last_match(_RE_BACKBONE, content)
    if m:
        result['backbone'] = m.group(1)

    m = _last_match(_RE_INIT_CLS, content)
    if m:
        result['init_cls'] = int(m.group(1))

    m = _last_match(_RE_INCREMENT, content)
    if m:
        result['increment'] = int(m.group(1))

    # --- Accuracy curves (take final curve in this seed block) ---
    m = _last_match(_RE_ACC_TOP1, content)
    if m:
        values = _parse_float_list(m.group(1))
        result['top1_curve'] = values
        if values:
            result['final_top1'] = values[-1]

    m = _last_match(_RE_ACC_TOP5, content)
    if m:
        values = _parse_float_list(m.group(1))
        result['top5_curve'] = values
        if values:
            result['final_top5'] = values[-1]

    # --- Average accuracy (take final average in this seed block) ---
    m = _last_match(_RE_AVG_ACC, content)
    if m:
        result['avg_accuracy'] = float(m.group(1))

    # --- Forgetting ---
    matches = _RE_FORGETTING.findall(content)
    if matches:
        result['forgetting'] = _safe_float(matches[-1])

    # --- Per-task accuracy (from grouped output lines) ---
    _RE_GROUPED = re.compile(r"HC-SOINN: \{([^}]+)\}")
    grouped_matches = _RE_GROUPED.findall(content)
    if grouped_matches:
        last_grouped = grouped_matches[-1]
        per_task: Dict[str, float] = {}
        for key, value in re.findall(
            r"'([^']+)': (?:np\.float64\()?([-\d.eE+]+)(?:\))?",
            last_grouped,
        ):
            per_task[key] = float(value)
        result['per_task_accuracy'] = per_task

    # --- LifeTopoDict diagnostics ---
    last_diag_pos = content.rfind('[LifeTopoDict] Diagnostics:')
    if last_diag_pos >= 0:
        diag_line = content[last_diag_pos:last_diag_pos + 500]

        m = _RE_DIAG_PAD.search(diag_line)
        if m:
            result['PAD'] = _safe_float(m.group(1))

        m = _RE_DIAG_EFFRANK.search(diag_line)
        if m:
            result['EffRank'] = _safe_float(m.group(1))

        m = _RE_DIAG_GTE.search(diag_line)
        if m:
            result['GTE'] = _safe_float(m.group(1))

        m = _RE_DIAG_ATOMS.search(diag_line)
        if m:
            result['atom_count'] = int(m.group(1))

        m = _RE_DIAG_PROTECTED.search(diag_line)
        if m:
            result['protected_pct'] = float(m.group(1))

        m = _RE_DIAG_INACTIVE.search(diag_line)
        if m:
            result['inactive_pct'] = float(m.group(1))

        m = _RE_DIAG_INACTIVE_NODES.search(diag_line)
        if m:
            result['inactive_node_pct'] = float(m.group(1))

        m = _RE_DIAG_FILTERED_INACTIVE_NODES.search(diag_line)
        if m:
            result['filtered_inactive_node_pct'] = float(m.group(1))

        m = _RE_DIAG_RESIDUAL.search(diag_line)
        if m:
            result['avg_residual'] = _safe_float(m.group(1))

    memory_pos = max(
        content.rfind('[LifeTopoDict] Memory:'),
        content.rfind('[HC-SOINN] Memory:'),
    )
    if memory_pos >= 0:
        memory_line = content[memory_pos:memory_pos + 500]
        for key, value in _RE_MEMORY_VALUE.findall(memory_line):
            result[f'memory_{key}_mb'] = float(value)

    edge_pos = content.rfind('[LifeTopoDict] Edge scoring:')
    if edge_pos >= 0:
        edge_line = content[edge_pos:edge_pos + 500]
        m = _RE_EDGE_USE_RATE.search(edge_line)
        if m:
            result['edge_use_rate'] = _safe_float(m.group(1))
        m = _RE_EDGE_CLASS_USE_RATE.search(edge_line)
        if m:
            result['edge_class_use_rate'] = _safe_float(m.group(1))
        m = _RE_EDGE_MARGIN.search(edge_line)
        if m:
            result['edge_margin_contribution'] = _safe_float(m.group(1))
        m = _RE_EDGE_RISK.search(edge_line)
        if m:
            result['edge_risk_penalty'] = _safe_float(m.group(1))
        m = _RE_EDGE_ADJUSTMENT.search(edge_line)
        if m:
            result['edge_score_adjustment'] = _safe_float(m.group(1))

    fallback_pos = content.rfind('[LifeTopoDict] Raw fallback:')
    if fallback_pos >= 0:
        fallback_line = content[fallback_pos:].splitlines()[0]
        for key, value in _RE_RAW_FALLBACK_VALUE.findall(fallback_line):
            parsed = _safe_float(value)
            if parsed is not None:
                result[f'raw_fallback_{key}'] = parsed

    raw_aux_pos = content.rfind('[LifeTopoDict] Raw auxiliary:')
    if raw_aux_pos >= 0:
        raw_aux_line = content[raw_aux_pos:].splitlines()[0]
        for key, value in _RE_RAW_AUXILIARY_VALUE.findall(raw_aux_line):
            parsed = _safe_float(value)
            if parsed is not None:
                result[f'raw_auxiliary_{key}'] = parsed

    gate_fit_pos = content.rfind('[LifeTopoDict] Raw fallback gate fit:')
    if gate_fit_pos >= 0:
        gate_fit_line = content[gate_fit_pos:].splitlines()[0]
        for key, value in _RE_RAW_FALLBACK_GATE_VALUE.findall(gate_fit_line):
            parsed = _safe_float(value)
            if parsed is not None:
                result[f'raw_fallback_gate_{key}'] = parsed

    atom_gate_pos = content.rfind('[LifeTopoDict] Atom conflict gate fit:')
    if atom_gate_pos >= 0:
        atom_gate_line = content[atom_gate_pos:].splitlines()[0]
        for key, value in _RE_ATOM_CONFLICT_GATE_VALUE.findall(atom_gate_line):
            parsed = _safe_float(value)
            if parsed is not None:
                result[f'atom_conflict_gate_{key}'] = parsed

    score_bias_pos = content.rfind('[LifeTopoDict] Score bias fit:')
    if score_bias_pos >= 0:
        score_bias_line = content[score_bias_pos:].splitlines()[0]
        for key, value in _RE_SCORE_BIAS_VALUE.findall(score_bias_line):
            parsed = _safe_float(value)
            if parsed is not None:
                result[f'score_bias_{key}'] = parsed

    pair_margin_pos = content.rfind('[LifeTopoDict] Pair margin fit:')
    if pair_margin_pos >= 0:
        pair_margin_line = content[pair_margin_pos:].splitlines()[0]
        for key, value in _RE_PAIR_MARGIN_VALUE.findall(pair_margin_line):
            parsed = _safe_float(value)
            if parsed is not None:
                result[f'pair_margin_{key}'] = parsed

    topology_reliability_pos = content.rfind('[LifeTopoDict] Topology reliability arbitration:')
    if topology_reliability_pos >= 0:
        topology_reliability_line = content[topology_reliability_pos:].splitlines()[0]
        for key, value in _RE_TOPOLOGY_RELIABILITY_VALUE.findall(topology_reliability_line):
            parsed = _safe_float(value)
            if parsed is not None:
                result[f'topology_reliability_{key}'] = parsed

    class_residual_repair_pos = content.rfind('[LifeTopoDict] Class residual repair:')
    if class_residual_repair_pos >= 0:
        class_residual_repair_line = content[class_residual_repair_pos:].splitlines()[0]
        for key, value in _RE_CLASS_RESIDUAL_REPAIR_VALUE.findall(class_residual_repair_line):
            parsed = _safe_float(value)
            if parsed is not None:
                result[f'class_residual_repair_{key}'] = parsed

    return result


def parse_log_file_rows(log_path: str) -> List[Dict[str, object]]:
    """Parse a log file and return one result row per seed block.

    Args:
        log_path: Path to the ``.log`` file.

    Returns:
        List of result dicts.  Missing metrics are omitted.
    """
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: could not read {log_path}: {e}", file=sys.stderr)
        return [{'log_file': os.path.basename(log_path)}]

    meta = _global_metadata(content)
    rows = [
        _parse_log_block(log_path, block, idx)
        for idx, block in enumerate(_seed_blocks(content))
    ]
    for row in rows:
        for key, value in meta.items():
            row.setdefault(key, value)
    return [row for row in rows if len(row) > 1]


def parse_log_file(log_path: str) -> Dict[str, object]:
    """Parse a log file and return the final seed row for compatibility."""
    rows = parse_log_file_rows(log_path)
    return rows[-1] if rows else {'log_file': os.path.basename(log_path)}


def collect_from_directory(
    log_dir: str,
    recursive: bool = False,
) -> List[Dict[str, object]]:
    """Collect results from all ``.log`` files in a directory.

    Args:
        log_dir: Directory to scan.
        recursive: If True, scan subdirectories too.

    Returns:
        List of result dicts, one per log file.
    """
    results: List[Dict[str, object]] = []

    if recursive:
        for root, dirs, files in os.walk(log_dir):
            for fname in sorted(files):
                if fname.endswith('.log'):
                    fpath = os.path.join(root, fname)
                    results.extend(parse_log_file_rows(fpath))
    else:
        for fname in sorted(os.listdir(log_dir)):
            if fname.endswith('.log'):
                fpath = os.path.join(log_dir, fname)
                results.extend(parse_log_file_rows(fpath))

    return results


def write_csv(
    results: List[Dict[str, object]],
    output_path: str,
) -> None:
    """Write collected results to a CSV file.

    Each row is one log file.  Columns include config params, accuracy
    metrics, forgetting, and LifeTopoDict diagnostics.
    """
    if not results:
        print("No results to write.", file=sys.stderr)
        return

    # Determine columns from all results
    # Priority columns first, then alphabetical for the rest
    priority_cols = [
        'log_file', 'prefix', 'dataset', 'seed', 'backbone',
        'init_cls', 'increment',
        'avg_accuracy', 'final_top1', 'final_top5', 'forgetting',
        'PAD', 'EffRank', 'GTE', 'atom_count', 'protected_pct',
        'inactive_pct', 'inactive_node_pct', 'filtered_inactive_node_pct',
        'avg_residual',
        'memory_compact_mb', 'memory_actual_mb',
        'memory_node_centers_mb', 'memory_atoms_mb', 'memory_coeffs_mb',
        'memory_edges_mb', 'memory_edge_rel_mb', 'memory_edge_reliability_mb',
        'memory_caches_mb', 'memory_buffers_mb', 'memory_frozen_mb',
        'memory_fallback_mb', 'memory_fallback_gate_mb',
        'memory_residual_penalty_mb', 'memory_residual_repair_mb',
        'memory_raw_aux_mb',
        'memory_pair_margin_mb',
        'memory_topology_reliability_mb',
        'memory_class_residual_repair_mb',
        'edge_use_rate', 'edge_class_use_rate', 'edge_margin_contribution',
        'edge_risk_penalty', 'edge_score_adjustment',
        'raw_auxiliary_enabled', 'raw_auxiliary_penalty',
        'raw_auxiliary_cache_nodes', 'raw_auxiliary_cache_classes',
        'raw_auxiliary_available', 'raw_auxiliary_change',
        'raw_auxiliary_compact_acc', 'raw_auxiliary_final_acc',
        'raw_auxiliary_benefit', 'raw_auxiliary_harm', 'raw_auxiliary_net',
        'raw_fallback_rate', 'raw_fallback_compact_acc',
        'raw_fallback_fallback_acc', 'raw_fallback_final_acc',
        'raw_fallback_oracle_improvable',
        'raw_fallback_gate_samples', 'raw_fallback_gate_positives',
        'raw_fallback_gate_gain', 'raw_fallback_gate_fallback_rate',
        'raw_fallback_gate_disabled',
        'atom_conflict_gate_samples', 'atom_conflict_gate_errors',
        'atom_conflict_gate_gain', 'atom_conflict_gate_gate_rate',
        'atom_conflict_gate_deployed', 'atom_conflict_gate_disabled',
        'score_bias_samples', 'score_bias_compact_acc',
        'score_bias_calib_acc', 'score_bias_gain',
        'score_bias_bias', 'score_bias_disabled',
        'pair_margin_samples', 'pair_margin_deployed',
        'pair_margin_gain', 'pair_margin_gate_rate',
        'pair_margin_disabled',
        'topology_reliability_enabled',
        'topology_reliability_margin_cap',
        'topology_reliability_min_gap',
        'topology_reliability_penalty',
        'topology_reliability_eval_gate_rate',
        'topology_reliability_eval_change_rate',
        'topology_reliability_eval_compact_acc',
        'topology_reliability_eval_final_acc',
        'topology_reliability_benefit',
        'topology_reliability_harm',
        'class_residual_repair_enabled',
        'class_residual_repair_stored',
        'class_residual_repair_updated',
        'class_residual_repair_strength',
        'class_residual_repair_min_nodes',
        'class_residual_repair_vector_norm_mean',
        'class_residual_repair_vector_norm_max',
    ]

    all_keys: List[str] = []
    seen_keys = set()
    for key in priority_cols:
        for r in results:
            if key in r and key not in seen_keys:
                all_keys.append(key)
                seen_keys.add(key)
                break

    # Add per-task accuracy columns
    per_task_keys: List[str] = []
    for r in results:
        pt = r.get('per_task_accuracy')
        if isinstance(pt, dict):
            for k in sorted(pt.keys()):
                col_name = f'acc_{k}'
                if col_name not in seen_keys:
                    per_task_keys.append(col_name)
                    seen_keys.add(col_name)

    # Add remaining keys
    for r in results:
        for key in sorted(r.keys()):
            if key not in seen_keys and key not in ('top1_curve', 'top5_curve', 'per_task_accuracy'):
                all_keys.append(key)
                seen_keys.add(key)

    all_keys.extend(per_task_keys)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()

        for r in results:
            row = dict(r)
            # Flatten per_task_accuracy
            pt = r.get('per_task_accuracy')
            if isinstance(pt, dict):
                for k, v in pt.items():
                    row[f'acc_{k}'] = v

            # Convert lists to string representation
            for key in ('top1_curve', 'top5_curve'):
                if key in row and isinstance(row[key], list):
                    row[key] = ';'.join(f'{x:.4f}' for x in row[key])

            writer.writerow({k: row.get(k, '') for k in all_keys})

    print(f"Wrote {len(results)} result(s) to {output_path}")


# ====================================================================== #
# CLI entry point                                                         #
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description='Collect LifeTopoDict experiment results from log files into CSV.'
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Log file(s) or directory(ies) containing .log files.',
    )
    parser.add_argument(
        '-o', '--output',
        default='collected_results.csv',
        help='Output CSV file path (default: collected_results.csv).',
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Recursively search subdirectories for .log files.',
    )

    args = parser.parse_args()

    all_results: List[Dict[str, object]] = []

    for path in args.paths:
        if os.path.isfile(path):
            all_results.extend(parse_log_file_rows(path))
        elif os.path.isdir(path):
            all_results.extend(collect_from_directory(path, recursive=args.recursive))
        else:
            print(f"Warning: {path} not found, skipping.", file=sys.stderr)

    if not all_results:
        print("No log files found.", file=sys.stderr)
        sys.exit(1)

    write_csv(all_results, args.output)


if __name__ == '__main__':
    main()
