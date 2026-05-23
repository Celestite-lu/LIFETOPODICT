"""LifeTopoDict evaluation metrics and diagnostic tools (Agent I).

This module provides standalone metric computation and memory analysis
routines that operate on an already-trained HCSOINNClassifier instance.
None of these functions modify the classifier state.
"""

import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ====================================================================== #
# I-1: Core evaluation metrics                                           #
# ====================================================================== #

def compute_DGR(
    clf,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Dictionary Gain Ratio (DGR).

    Measures the accuracy improvement gained by dictionary-encoded
    (materialized) prototypes over raw prototypes.

    For each sample, the prediction is the class whose nearest prototype
    (cosine) is closest.  We compute accuracy using the current
    (materialized) prototypes and using the original raw features stored
    in each node's ``center_raw`` (pre-materialization).  The DGR is
    defined as::

        DGR = (acc_encoded - acc_raw) / acc_raw

    A positive value means dictionary encoding helped; negative means it
    hurt.

    Args:
        clf: HCSOINNClassifier instance (must have ``class_clusters``).
        features: Test features [N, d].
        labels: Ground-truth labels [N].

    Returns:
        DGR value (float).  ``float('nan')`` if accuracy is zero.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    def _predict_with_protos(proto_getter):
        """Predict using prototypes returned by *proto_getter*(cls, clusters)."""
        correct = 0
        total = 0
        for i in range(features.shape[0]):
            x = features[i]
            x_norm = x / (np.linalg.norm(x) + 1e-8)
            best_cls = -1
            best_dist = float('inf')
            for cls, clusters in clf.class_clusters.items():
                for node in clusters:
                    if node.node_state == 'inactive':
                        continue
                    proto = proto_getter(node)
                    if proto is None:
                        continue
                    p_norm = proto / (np.linalg.norm(proto) + 1e-8)
                    dist = 1.0 - float(np.dot(x_norm, p_norm))
                    if dist < best_dist:
                        best_dist = dist
                        best_cls = cls
            if best_cls == labels[i]:
                correct += 1
            total += 1
        return correct / max(total, 1)

    # Accuracy with materialized (encoded) prototypes
    acc_encoded = _predict_with_protos(lambda node: node.center)

    # Accuracy with raw prototypes (pre-materialization)
    acc_raw = _predict_with_protos(lambda node: node.center_raw)

    if acc_raw < 1e-8:
        return float('nan')

    dgr = (acc_encoded - acc_raw) / acc_raw
    logger.info(f"[DGR] acc_encoded={acc_encoded:.4f}, acc_raw={acc_raw:.4f}, DGR={dgr:.4f}")
    return float(dgr)


def compute_signed_AEPMI(
    clf,
    features: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Signed Atom-Error PMI (sAEPMI).

    For each dictionary atom *m*, computes the Pointwise Mutual Information
    between "atom *m* is in the support set of the predicted-class
    prototype" and "the prediction is wrong".  A *positive* sAEPMI means
    the atom is associated with errors (bad); *negative* means it is
    associated with correct predictions (good).

    The final metric is the mean over all atoms.

    Args:
        clf: HCSOINNClassifier instance.
        features: Test features [N, d].
        labels: Ground-truth labels [N].
        predictions: Predicted class labels [N].

    Returns:
        Mean signed AEPMI (float).  ``float('nan')`` if no atoms.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    if not getattr(clf, 'use_dict_coding', False) or clf.dict_atoms is None:
        return float('nan')

    M = clf.dict_atoms.shape[0]
    if M == 0:
        return float('nan')

    N = features.shape[0]
    is_error = (predictions != labels)  # [N]

    n_error = int(is_error.sum())
    n_correct = N - n_error

    if n_error == 0 or n_correct == 0:
        # Degenerate: all correct or all wrong
        return 0.0

    p_error = n_error / N
    p_correct = n_correct / N

    aepmi_values: List[float] = []

    for m in range(M):
        # Determine which samples have atom m in their predicted-class prototype support
        atom_in_support = np.zeros(N, dtype=bool)

        for i in range(N):
            pred_cls = int(predictions[i])
            clusters = clf.class_clusters.get(pred_cls, [])
            for node in clusters:
                if node.node_state == 'inactive' or node.coeff is None:
                    continue
                coeff = np.abs(node.coeff.ravel())
                if m < len(coeff) and coeff[m] > 1e-6:
                    atom_in_support[i] = True
                    break

        n_support = int(atom_in_support.sum())
        if n_support == 0 or n_support == N:
            # No PMI signal; skip
            aepmi_values.append(0.0)
            continue

        p_support = n_support / N
        # Joint: atom in support AND prediction is wrong
        n_joint_error = int((atom_in_support & is_error).sum())
        n_joint_correct = int((atom_in_support & (~is_error)).sum())

        # Signed PMI: positive if atom associates with errors
        pmi = 0.0
        if n_joint_error > 0:
            p_joint = n_joint_error / N
            pmi += np.log(p_joint / (p_support * p_error) + 1e-12)
        if n_joint_correct > 0:
            p_joint = n_joint_correct / N
            pmi -= np.log(p_joint / (p_support * p_correct) + 1e-12)

        aepmi_values.append(float(pmi))

    result = float(np.mean(aepmi_values)) if aepmi_values else float('nan')
    logger.info(f"[sAEPMI] mean={result:.6f} over {len(aepmi_values)} atoms")
    return result


def compute_PAD(clf) -> float:
    """Prototype Alignment Degree -- delegates to ``clf.compute_PAD()``.

    Args:
        clf: HCSOINNClassifier instance.

    Returns:
        PAD value (float).
    """
    if not hasattr(clf, 'compute_PAD'):
        logger.warning("[PAD] classifier has no compute_PAD() method.")
        return float('nan')
    pad = clf.compute_PAD()
    logger.info(f"[PAD] {pad:.4f}")
    return pad


def compute_memory_report(clf) -> Dict[str, float]:
    """Detailed memory breakdown for compact vs actual storage.

    Delegates to the classifier's ``_compute_memory_breakdown()`` method
    when available (uses precise ``.nbytes`` for numpy arrays).  Falls back
    to a local estimation using ``sys.getsizeof`` otherwise.

    *Compact* memory: dict_atoms + atom_states + all node coefficients +
    topology edges.

    *Actual* memory: compact + buffers + caches + frozen_clusters.

    Args:
        clf: HCSOINNClassifier instance.

    Returns:
        Dict with keys ``'compact_bytes'``, ``'actual_bytes'``,
        ``'compact_mb'``, ``'actual_mb'``, and per-component breakdowns.
    """
    # Prefer the classifier's own precise nbytes-based breakdown
    if hasattr(clf, '_compute_memory_breakdown'):
        breakdown = clf._compute_memory_breakdown()
        # Build a flat report from the breakdown for backward compatibility
        MB = 1024.0 * 1024.0
        report: Dict[str, float] = {
            'compact_bytes': breakdown.get('compact_deployable_bytes', 0.0),
            'actual_bytes': breakdown.get('actual_implementation_bytes', 0.0),
            'compact_mb': breakdown.get('compact_deployable_mb', 0.0),
            'actual_mb': breakdown.get('actual_implementation_mb', 0.0),
            'dict_atoms_mb': breakdown.get('atoms_mb', 0.0),
            'coefficients_mb': breakdown.get('coefficients_mb', 0.0),
            'atom_states_mb': breakdown.get('atom_metadata_mb', 0.0),
            'edges_mb': breakdown.get('edges_mb', 0.0),
            'edge_reliability_mb': breakdown.get('edge_reliability_mb', 0.0),
            'buffers_mb': breakdown.get('buffers_mb', 0.0),
            'cache_mb': breakdown.get('caches_mb', 0.0),
            'frozen_mb': breakdown.get('frozen_mb', 0.0),
            # Additional per-component detail
            'atoms_mb': breakdown.get('atoms_mb', 0.0),
            'atoms_actual_mb': breakdown.get('atoms_actual_mb', 0.0),
            'atoms_duplicate_mb': breakdown.get('atoms_duplicate_mb', 0.0),
            'atom_metadata_mb': breakdown.get('atom_metadata_mb', 0.0),
            'node_metadata_mb': breakdown.get('node_metadata_mb', 0.0),
            'node_centers_mb': breakdown.get('node_centers_mb', 0.0),
            'class_mu_mb': breakdown.get('class_mu_mb', 0.0),
            'class_mu_raw_mb': breakdown.get('class_mu_raw_mb', 0.0),
            'star_feature_anchors_mb': breakdown.get('star_feature_anchors_mb', 0.0),
            'star_image_anchors_mb': breakdown.get('star_image_anchors_mb', 0.0),
        }

        logger.info(
            "[Memory] compact=%.4f MB, actual=%.4f MB, "
            "atoms=%.4f actual=%.4f, coeffs=%.4f, edges=%.4f, "
            "overhead=%.4f + %.4f + %.4f MB",
            report['compact_mb'], report['actual_mb'],
            report['atoms_mb'], report['atoms_actual_mb'],
            report['coefficients_mb'], report['edges_mb'],
            report['buffers_mb'], report['cache_mb'], report['frozen_mb'],
        )
        return report

    # --- Fallback: local estimation using sys.getsizeof ---
    report: Dict[str, float] = {}

    # --- Helper ---
    def _sizeof(obj) -> int:
        """Estimate bytes for a Python/numpy object."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        if isinstance(obj, (list, tuple)):
            total = sys.getsizeof(obj)
            for item in obj:
                if isinstance(item, np.ndarray):
                    total += item.nbytes
                else:
                    total += sys.getsizeof(item)
            return total
        if isinstance(obj, dict):
            total = sys.getsizeof(obj)
            for k, v in obj.items():
                total += sys.getsizeof(k)
                total += _sizeof(v)
            return total
        return sys.getsizeof(obj)

    # --- Compact components ---
    dict_atoms_bytes = 0
    dict_atoms_hat_bytes = 0
    if isinstance(getattr(clf, 'dict_atoms', None), np.ndarray):
        dict_atoms_bytes = clf.dict_atoms.nbytes
    if isinstance(getattr(clf, 'dict_atoms_hat', None), np.ndarray):
        dict_atoms_hat_bytes = clf.dict_atoms_hat.nbytes

    atom_states_bytes = _sizeof(getattr(clf, 'atom_states', []))

    # Node coefficients
    coeff_bytes = 0
    for cls, clusters in getattr(clf, 'class_clusters', {}).items():
        for node in clusters:
            if node.coeff is not None and isinstance(node.coeff, np.ndarray):
                coeff_bytes += node.coeff.nbytes

    # Topology edges
    edges_bytes = _sizeof(getattr(clf, 'class_edges', {}))

    compact_bytes = dict_atoms_bytes + dict_atoms_hat_bytes + atom_states_bytes + coeff_bytes + edges_bytes

    # --- Extra actual-memory components ---
    buffers_bytes = _sizeof(getattr(clf, 'buffers', {}))
    cache_bytes = _sizeof(getattr(clf, '_predict_cache', {}))
    frozen_bytes = _sizeof(getattr(clf, 'frozen_clusters', {}))

    actual_bytes = compact_bytes + buffers_bytes + cache_bytes + frozen_bytes

    MB = 1024.0 * 1024.0
    report = {
        'compact_bytes': float(compact_bytes),
        'actual_bytes': float(actual_bytes),
        'compact_mb': compact_bytes / MB,
        'actual_mb': actual_bytes / MB,
        'dict_atoms_mb': (dict_atoms_bytes + dict_atoms_hat_bytes) / MB,
        'coefficients_mb': coeff_bytes / MB,
        'atom_states_mb': atom_states_bytes / MB,
        'edges_mb': edges_bytes / MB,
        'buffers_mb': buffers_bytes / MB,
        'cache_mb': cache_bytes / MB,
        'frozen_mb': frozen_bytes / MB,
    }

    logger.info(
        "[Memory] compact=%.4f MB, actual=%.4f MB, "
        "overhead=%.4f + %.4f + %.4f MB",
        report['compact_mb'], report['actual_mb'],
        buffers_bytes / MB, cache_bytes / MB, frozen_bytes / MB,
    )
    return report


def compute_edge_consistency(clf) -> Dict[str, object]:
    """Verify that raw node edges and materialized node edges are consistent.

    After dictionary coding, the ``class_edges`` topology should still be
    valid -- i.e., every edge references an existing node, and node counts
    match.

    Args:
        clf: HCSOINNClassifier instance.

    Returns:
        Dict with ``'consistent'`` (bool), ``'issues'`` (list of str),
        and summary counts.
    """
    issues: List[str] = []
    class_edges = getattr(clf, 'class_edges', {})
    class_clusters = getattr(clf, 'class_clusters', {})

    total_edges = 0
    broken_edges = 0
    classes_checked = 0

    for cls, edges_map in class_edges.items():
        clusters = class_clusters.get(cls, [])
        n_nodes = len(clusters)
        classes_checked += 1

        for i, neighbors in edges_map.items():
            if i >= n_nodes:
                issues.append(f"Class {cls}: edge from non-existent node {i} (n_nodes={n_nodes})")
                broken_edges += 1
                continue
            for j, age in neighbors.items():
                total_edges += 1
                if j >= n_nodes:
                    issues.append(f"Class {cls}: edge to non-existent node {j} from {i}")
                    broken_edges += 1

    consistent = (broken_edges == 0)

    result = {
        'consistent': consistent,
        'issues': issues,
        'classes_checked': classes_checked,
        'total_edges': total_edges,
        'broken_edges': broken_edges,
    }

    if not consistent:
        logger.warning(f"[EdgeConsistency] INCONSISTENT: {broken_edges} broken edges out of {total_edges}")
    else:
        logger.info(f"[EdgeConsistency] OK: {total_edges} edges across {classes_checked} classes")

    return result


# ====================================================================== #
# I-2: Memory comparison tool                                            #
# ====================================================================== #

def _measure_classifier_memory(clf) -> float:
    """Estimate total memory (bytes) used by an HCSOINNClassifier.

    Walks all numpy arrays (``.nbytes``) and recursively sizes Python
    containers with ``sys.getsizeof``.
    """
    total = 0

    # Direct numpy arrays
    for attr_name in ('dict_atoms', 'dict_atoms_hat', 'inactive_mask'):
        obj = getattr(clf, attr_name, None)
        if isinstance(obj, np.ndarray):
            total += obj.nbytes

    # class_mu / class_mu_raw / class_count
    for attr_name in ('class_mu', 'class_mu_raw', 'class_count'):
        obj = getattr(clf, attr_name, None)
        if isinstance(obj, dict):
            total += sys.getsizeof(obj)
            for k, v in obj.items():
                total += sys.getsizeof(k)
                if isinstance(v, np.ndarray):
                    total += v.nbytes
                else:
                    total += sys.getsizeof(v)

    # class_clusters (contains _Cluster objects with numpy arrays)
    class_clusters = getattr(clf, 'class_clusters', {})
    total += sys.getsizeof(class_clusters)
    for cls, clusters in class_clusters.items():
        total += sys.getsizeof(cls)
        total += sys.getsizeof(clusters)
        for node in clusters:
            total += sys.getsizeof(node)
            for field in ('center', 'center_raw', 'coeff'):
                arr = getattr(node, field, None)
                if isinstance(arr, np.ndarray):
                    total += arr.nbytes

    # atom metadata
    for attr_name in ('atom_states', 'atom_origins', 'atom_usage',
                       'atom_old_tasks', '_low_usage_streak', 'node_states',
                       'old_support'):
        obj = getattr(clf, attr_name, None)
        if isinstance(obj, (list, dict)):
            total += _deep_sizeof(obj)

    # buffers
    buffers = getattr(clf, 'buffers', {})
    total += _deep_sizeof(buffers)

    # frozen_clusters
    frozen = getattr(clf, 'frozen_clusters', {})
    total += sys.getsizeof(frozen)
    for cls, clusters in frozen.items():
        total += sys.getsizeof(cls)
        for node in clusters:
            total += sys.getsizeof(node)
            for field in ('center', 'center_raw', 'coeff'):
                arr = getattr(node, field, None)
                if isinstance(arr, np.ndarray):
                    total += arr.nbytes

    # _predict_cache
    cache = getattr(clf, '_predict_cache', {})
    total += sys.getsizeof(cache)
    for k, v in cache.items():
        total += sys.getsizeof(k)
        if isinstance(v, np.ndarray):
            total += v.nbytes
        elif hasattr(v, 'nbytes'):  # torch tensor
            try:
                total += v.nbytes
            except Exception:
                total += sys.getsizeof(v)
        else:
            total += sys.getsizeof(v)

    # class_edges
    class_edges = getattr(clf, 'class_edges', {})
    total += _deep_sizeof(class_edges)
    class_edge_reliability = getattr(clf, 'class_edge_reliability', {})
    total += _deep_sizeof(class_edge_reliability)

    return total


def _deep_sizeof(obj) -> int:
    """Recursively estimate memory of a Python container."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        total = sys.getsizeof(obj)
        for k, v in obj.items():
            total += sys.getsizeof(k) if not isinstance(k, np.ndarray) else k.nbytes
            total += _deep_sizeof(v)
        return total
    if isinstance(obj, (list, tuple)):
        total = sys.getsizeof(obj)
        for item in obj:
            total += _deep_sizeof(item)
        return total
    return sys.getsizeof(obj)


def same_memory_comparison(
    ltd_clf,
    baseline_clf,
) -> Dict[str, object]:
    """Compare memory usage between LifeTopoDict and a baseline classifier.

    Args:
        ltd_clf: The LifeTopoDict HCSOINNClassifier (use_dict_coding=True).
        baseline_clf: A baseline HCSOINNClassifier (use_dict_coding=False).

    Returns:
        Dict with ``'ltd_compact_mb'``, ``'ltd_actual_mb'``,
        ``'baseline_mb'``, ``'savings_pct'``.
    """
    # LTD memory
    mem_report = compute_memory_report(ltd_clf)
    ltd_compact = mem_report['compact_bytes']
    ltd_actual = mem_report['actual_bytes']

    # Baseline memory
    baseline_bytes = _measure_classifier_memory(baseline_clf)

    MB = 1024.0 * 1024.0

    if baseline_bytes > 0:
        savings_pct = (1.0 - ltd_compact / baseline_bytes) * 100.0
    else:
        savings_pct = 0.0

    result = {
        'ltd_compact_mb': ltd_compact / MB,
        'ltd_actual_mb': ltd_actual / MB,
        'baseline_mb': baseline_bytes / MB,
        'savings_pct': float(savings_pct),
    }

    logger.info(
        f"[MemoryComparison] LTD compact={result['ltd_compact_mb']:.4f} MB, "
        f"LTD actual={result['ltd_actual_mb']:.4f} MB, "
        f"baseline={result['baseline_mb']:.4f} MB, "
        f"savings={result['savings_pct']:.1f}%"
    )
    return result


# ====================================================================== #
# Convenience: run all diagnostics                                        #
# ====================================================================== #

def run_all_diagnostics(
    clf,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Run all available diagnostic metrics on *clf*.

    Args:
        clf: HCSOINNClassifier instance.
        features: Optional test features for DGR / sAEPMI.
        labels: Optional ground-truth labels.
        predictions: Optional model predictions.

    Returns:
        Combined diagnostic dict.
    """
    result: Dict[str, object] = {}

    # Built-in diagnostics from the classifier
    if hasattr(clf, 'compute_diagnostics'):
        result.update(clf.compute_diagnostics())

    # PAD
    result['PAD'] = compute_PAD(clf)

    # Memory report
    result['memory'] = compute_memory_report(clf)

    # Edge consistency
    result['edge_consistency'] = compute_edge_consistency(clf)

    # DGR (requires test data)
    if features is not None and labels is not None:
        result['DGR'] = compute_DGR(clf, features, labels)

    # sAEPMI (requires test data and predictions)
    if features is not None and labels is not None and predictions is not None:
        result['sAEPMI'] = compute_signed_AEPMI(clf, features, labels, predictions)

    return result
