"""Core component."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import time
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import logging
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import random


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v.astype(np.float32, copy=True)
    return (v / norm).astype(np.float32, copy=True)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a_norm, b_norm))


class _Cluster:
    def __init__(self, center: np.ndarray, count: int, center_raw: Optional[np.ndarray] = None,
                 coeff: Optional[np.ndarray] = None, residual: float = 0.0,
                 node_state: str = 'plastic',
                 dict_recon_raw: Optional[np.ndarray] = None):
        """Handle init.

        Args:
            center: Normalized cluster center.
            count: Number of samples assigned to this cluster.
            center_raw: Raw (unnormalized) cluster center. Defaults to a copy of *center*.
            coeff: Sparse coefficient vector (1, M) on the shared dictionary.
                   ``None`` means the node has not been dictionary-coded yet.
            residual: Cosine reconstruction residual ``1 - cos_sim(v, D @ a)``.
            node_state: Lifecycle state — ``'protected'``, ``'plastic'``, or ``'inactive'``.
            dict_recon_raw: Raw dictionary reconstruction ``D @ a``.  This is
                kept separate from ``center_raw``, which remains the true raw
                cluster mean / sparse-coding target.
        """
        self.center = _normalize(center)
        self.count = int(count)
        if center_raw is not None:
            self.center_raw = center_raw.copy()
        else:
            self.center_raw = center.copy()
        # --- LifeTopoDict: dictionary-coding fields ---
        self.coeff: Optional[np.ndarray] = coeff          # (1, M) or None
        self.residual: float = float(residual)
        self.node_state: str = node_state  # 'protected' | 'plastic' | 'inactive'
        self.dict_recon_raw: Optional[np.ndarray] = (
            None if dict_recon_raw is None else dict_recon_raw.copy()
        )
            
def _hierarchical_cluster(feats_norm: np.ndarray, feats_raw: np.ndarray, target_k: int, linkage_method: str, distance_metric: str) -> List[_Cluster]:
    """Handle hierarchical cluster."""
    if feats_norm.shape[0] == 0:
        return []
    if feats_norm.shape[0] <= target_k:
        return [_Cluster(feats_norm[i], 1, center_raw=feats_raw[i]) for i in range(feats_norm.shape[0])]

    try:
        Z = linkage(feats_norm, method=linkage_method, metric=distance_metric)
        cluster_ids = fcluster(Z, t=target_k, criterion="maxclust")
    except Exception:
        return [_Cluster(feats_norm[i], 1, center_raw=feats_raw[i]) for i in range(feats_norm.shape[0])]

    clusters_map = defaultdict(list)
    clusters_raw_map = defaultdict(list)
    
    for i, cid in enumerate(cluster_ids):
        clusters_map[cid].append(feats_norm[i])
        clusters_raw_map[cid].append(feats_raw[i])

    out: List[_Cluster] = []
    for cid in clusters_map:
        vectors = np.stack(clusters_map[cid])
        vectors_raw = np.stack(clusters_raw_map[cid])
        
        center = vectors.mean(axis=0)
        center_raw = vectors_raw.mean(axis=0)
        
        out.append(_Cluster(center, vectors.shape[0], center_raw=center_raw))
    return out

def _merge_close_clusters(clusters: List[_Cluster], tau: float) -> List[_Cluster]:
    """Handle merge close clusters."""
    if len(clusters) <= 1:
        return clusters

    centers = np.stack([c.center for c in clusters], axis=0)
    dmat = cdist(centers, centers, metric="cosine")

    merged = [False] * len(clusters)
    new_clusters: List[_Cluster] = []

    for i in range(len(clusters)):
        if merged[i]:
            continue
        close_idxs = [i]
        for j in range(i + 1, len(clusters)):
            if dmat[i, j] <= tau:
                merged[j] = True
                close_idxs.append(j)
        total_count = sum(clusters[k].count for k in close_idxs)
        weighted_center = sum(clusters[k].center * clusters[k].count for k in close_idxs) / float(
            max(total_count, 1)
        )
        weighted_center_raw = sum(clusters[k].center_raw * clusters[k].count for k in close_idxs) / float(
            max(total_count, 1)
        )
        new_clusters.append(_Cluster(weighted_center, total_count, center_raw=weighted_center_raw))

    return new_clusters


def _match_protected_indices(
    protected_centers: Optional[List[np.ndarray]],
    clusters: List[_Cluster],
    threshold: float = 0.7,
) -> Optional[Set[int]]:
    """Greedily map old protected centers to a rebuilt cluster list."""
    if not protected_centers or not clusters:
        return None

    new_centers = np.stack([c.center for c in clusters], axis=0)
    matched: Set[int] = set()
    used_new: Set[int] = set()
    for old_center in protected_centers:
        old_norm = _normalize(old_center)
        sims = new_centers @ old_norm
        for used in used_new:
            sims[used] = -2.0
        best = int(np.argmax(sims))
        if float(sims[best]) > threshold:
            matched.add(best)
            used_new.add(best)

    return matched if matched else None


def _spherical_interpolate(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """Handle spherical interpolate."""
    v1_norm = _normalize(v1)
    v2_norm = _normalize(v2)
    
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    
    if abs(dot) > 0.9995:
        return _normalize((1 - t) * v1_norm + t * v2_norm)
    
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    result = w1 * v1_norm + w2 * v2_norm
    return _normalize(result)

def _simplified_soinn_on_clusters(
    cluster_centers: List[np.ndarray],     # Normalized
    cluster_centers_raw: List[np.ndarray], # Raw (Unnormalized)
    cluster_counts: List[int],
    ad: int = 20,
    lam: int = 20,
    threshold_scale: float = 0.5,
    max_iterations: int = 3,
    max_degree_for_removal: int = 1,
    protected_indices: Optional[Set[int]] = None,
) -> Tuple[List[_Cluster], Dict[int, Dict[int, int]]]:
    """Handle simplified soinn on clusters."""
    if len(cluster_centers) == 0:
        return [], {}
    
    nodes = [_normalize(c.copy()) for c in cluster_centers]
    nodes_raw = [c.copy() for c in cluster_centers_raw]
    
    win_counts = cluster_counts.copy()
    edges = defaultdict(dict)
    
    protected = set(protected_indices or set())

    n_nodes = len(nodes)
    if n_nodes >= 2:
        for i in range(n_nodes):
            dists = [_cosine_distance(nodes[i], nodes[j]) if i != j else float('inf') for j in range(n_nodes)]
            if len(dists) > 0:
                nearest_idx = np.argmin(dists)
                edges[i][nearest_idx] = 0
                edges[nearest_idx][i] = 0
    
    input_signals = list(zip(cluster_centers, cluster_centers_raw))
    
    # IMPORTANT (Determinism):
    # This refinement previously used python's global `random.shuffle(indices)`, which makes results
    # depend on the global RNG state of each worker process. Enabling/disabling STAR (or any other
    # feature) can consume RNG calls before `compress()`, changing the shuffle order and producing
    # different prototypes even on Task0 (where STAR hasn't aligned anything yet).
    #
    # Use a local RNG with a fixed seed so refinement order is reproducible and independent of
    # global RNG / multiprocessing for comparable experiments.
    _local_rng = np.random.RandomState(0)
    
    for iteration in range(max_iterations):
        indices = list(range(len(input_signals)))
        if len(indices) > 1:
            _local_rng.shuffle(indices)
        
        for t, idx in enumerate(indices, start=1):
            x, x_raw = input_signals[idx]
            x_norm = _normalize(x)
            
            if len(nodes) < 2: continue
            
            dists = np.array([_cosine_distance(x_norm, w) for w in nodes])
            sorted_idx = np.argsort(dists)
            s1, s2 = sorted_idx[0], sorted_idx[1]
            
            
            if s2 not in edges[s1]:
                edges[s1][s2] = 0
                edges[s2][s1] = 0
            
            for nbr in list(edges[s1].keys()):
                edges[s1][nbr] += 1
                if s1 in edges[nbr]: edges[nbr][s1] = edges[s1][nbr]
                if edges[s1][nbr] > ad:
                    del edges[s1][nbr]
                    if s1 in edges[nbr]: del edges[nbr][s1]
            
            eta1 = 1.0 / float(t + iteration * len(input_signals) + 1)
            eta2 = 1.0 / (100.0 * float(t + iteration * len(input_signals) + 1))
            
            if s1 not in protected:
                nodes[s1] = _spherical_interpolate(nodes[s1], x_norm, eta1)

                # w_raw += eta * (x_raw - w_raw)
                nodes_raw[s1] = nodes_raw[s1] + eta1 * (x_raw - nodes_raw[s1])
            
            for nbr in list(edges[s1].keys()):
                if nbr in protected:
                    continue
                nodes[nbr] = _spherical_interpolate(nodes[nbr], x_norm, eta2)
                nodes_raw[nbr] = nodes_raw[nbr] + eta2 * (x_raw - nodes_raw[nbr])
                
            win_counts[s1] += 1

        # ...
        # E-5: Protected nodes are not eligible for removal
        to_remove = [
            i for i in range(len(nodes))
            if len(edges[i]) <= max_degree_for_removal and i not in protected
        ]
        if len(to_remove) > 0 and len(nodes) - len(to_remove) >= 2:
            keep_mask = [i not in to_remove for i in range(len(nodes))]
            
            old_to_new = {}
            new_nodes, new_nodes_raw, new_counts = [], [], []
            for old_idx, keep in enumerate(keep_mask):
                if keep:
                    old_to_new[old_idx] = len(new_nodes)
                    new_nodes.append(nodes[old_idx])
                    new_nodes_raw.append(nodes_raw[old_idx])
                    new_counts.append(win_counts[old_idx])
            
            new_edges = defaultdict(dict)
            for old_i, nbrs in edges.items():
                if old_i in old_to_new:
                    new_i = old_to_new[old_i]
                    for old_j in nbrs:
                        if old_j in old_to_new:
                            new_edges[new_i][old_to_new[old_j]] = edges[old_i][old_j]
            
            protected = {old_to_new[i] for i in protected if i in old_to_new}
            nodes, nodes_raw, win_counts, edges = new_nodes, new_nodes_raw, new_counts, new_edges
            
            for i in range(len(nodes)):
                if len(edges[i]) == 0 and len(nodes) > 1:
                    dists = [_cosine_distance(nodes[i], nodes[j]) if i != j else float('inf') for j in range(len(nodes))]
                    nearest = np.argmin(dists)
                    edges[i][nearest] = 0
                    edges[nearest][i] = 0

    result = []
    final_edges_map = {}
    
    final_indices = [i for i in range(len(nodes)) if len(edges[i]) > 0]
    if not final_indices: final_indices = range(len(nodes))
    
    for new_idx, old_idx in enumerate(final_indices):
        result.append(_Cluster(nodes[old_idx], win_counts[old_idx], center_raw=nodes_raw[old_idx]))
        final_edges_map[new_idx] = {}
        for nbr, age in edges[old_idx].items():
             if nbr in final_indices:
                 new_nbr = final_indices.index(nbr)
                 final_edges_map[new_idx][new_nbr] = age
                 
    return result, final_edges_map

def _compress_class_worker(args):

    (cls, feats, target_k, tau_merge, linkage_method, distance_metric, max_prototypes,
     use_soinn_refinement, soinn_ad, soinn_lam, soinn_threshold_scale, soinn_max_iter,
     soinn_max_degree_for_removal, protected_centers) = args

    feats_raw = feats.astype(np.float32)
    feats_norm = feats_raw / (np.linalg.norm(feats_raw, axis=1, keepdims=True) + 1e-8)

    clusters = _hierarchical_cluster(feats_norm, feats_raw, target_k, linkage_method, distance_metric)
    hierarchical_count = len(clusters)

    soinn_edges = {}
    if use_soinn_refinement and len(clusters) > 1:
        protected_indices = _match_protected_indices(protected_centers, clusters)
        cluster_centers = [c.center for c in clusters]
        cluster_centers_raw = [c.center_raw for c in clusters]
        cluster_counts = [c.count for c in clusters]

        clusters, soinn_edges = _simplified_soinn_on_clusters(
            cluster_centers,
            cluster_centers_raw,
            cluster_counts,
            ad=soinn_ad,
            lam=soinn_lam,
            threshold_scale=soinn_threshold_scale,
            max_iterations=soinn_max_iter,
            max_degree_for_removal=soinn_max_degree_for_removal,
            protected_indices=protected_indices,
        )
    else:
        clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
    
    
    if len(clusters) > 0:
        raw_norms = [np.linalg.norm(c.center_raw) for c in clusters]
        avg_raw_norm = sum(raw_norms) / len(raw_norms)
        # print(f"DEBUG WORKER {cls}: Avg Raw Norm = {avg_raw_norm:.4f}")

    final_count = len(clusters)
    return cls, clusters, hierarchical_count, final_count, soinn_edges

class HCSOINNClassifier:
    """Handle init."""

    def __init__(
        self,
        max_prototypes_per_class: Optional[int] = 20,
        alpha: float = 0.5,
        tau_merge: float = 0.2,
        tau_reject: float = 2.0,
        linkage_method: str = "average",
        distance_metric: str = "cosine",
        use_soinn_refinement: bool = True,
        soinn_ad: int = 20,
        soinn_lam: int = 20,
        soinn_threshold_scale: float = 0.5,
        soinn_max_iter: int = 3,
        soinn_max_degree_for_removal: int = 1,
        coarse_topk: Optional[int] = None,
        enable_inference_profiling: bool = False,
        profile_sync_cuda: bool = True,
        # --- Lifecycle parameters (Agent E) ---
        lifecycle_theta_support: float = 0.5,
        lifecycle_theta_usage: float = 0.01,
        lifecycle_T_inactive: int = 3,
        lifecycle_min_alive_ratio: float = 0.70,
        lifecycle_protect_old_topk: bool = True,
        lifecycle_node_inactive_threshold: float = 1.0,
        # --- LifeTopoDict dictionary-coding parameters (Agent D) ---
        use_dict_coding: bool = False,
        dict_sparse_k: int = 5,
        dict_ridge_lambda: float = 0.1,
        drop_node_raw_after_dict_materialize: bool = True,
        # --- Dictionary growth parameters (Agent F) ---
        theta_residual: float = 0.3,
        max_growth_per_task: int = 5,
        # --- Direction 1: raw-node fallback gate ---
        use_raw_fallback_gate: bool = False,
        raw_fallback_per_class: int = 0,
        raw_fallback_select_by: str = "residual",
        fallback_gate_type: str = "rule",
        fallback_gate_margin_threshold: float = 0.05,
        fallback_gate_residual_threshold: float = 0.08,
        fallback_gate_random_rate: float = 0.10,
        fallback_memory_accounting: bool = True,
        fallback_calibration_samples_per_class: int = 0,
        fallback_gate_min_positives: int = 5,
        fallback_gate_min_calibration_gain: float = 0.001,
        fallback_gate_max_rate: float = 1.0,
        fallback_gate_tree_estimators: int = 160,
        fallback_gate_tree_max_depth: int = 2,
        fallback_gate_tree_min_samples_leaf: int = 12,
        fallback_gate_threshold_strategy: str = "max_net",
        fallback_gate_min_precision: float = 0.0,
        fallback_gate_min_selected: int = 0,
        fallback_gate_min_net_count: int = 0,
        fallback_candidate_rule: str = "none",
        fallback_gate_budget_rates: object = "0.002,0.005,0.01,0.02,0.04",
        fallback_gate_neutral_weight: float = 0.25,
        fallback_pair_table_mode: str = "none",
        fallback_pair_table_smoothing: float = 10.0,
        fallback_pair_table_lambda: float = 0.5,
        fallback_node_table_lambda: float = 0.5,
        enable_prediction_trace: bool = False,
        trace_split: str = "test",
        fallback_random_seed: int = 0,
    ) -> None:
        self.max_prototypes_per_class = None if max_prototypes_per_class is None else int(
            max_prototypes_per_class
        )
        self.alpha = float(alpha)
        self.tau_merge = float(tau_merge)
        self.tau_reject = float(tau_reject)
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        
        self.use_soinn_refinement = bool(use_soinn_refinement)
        self.soinn_ad = int(soinn_ad)
        self.soinn_lam = int(soinn_lam)
        self.soinn_threshold_scale = float(soinn_threshold_scale)
        self.soinn_max_iter = int(soinn_max_iter)
        self.soinn_max_degree_for_removal = int(soinn_max_degree_for_removal)

        # Two-stage coarse filtering: use NCM to shortlist candidate classes
        # before computing expensive sub-cluster distances.
        # None = disabled (evaluate all classes); int = number of NCM candidates.
        self.coarse_topk = None if coarse_topk is None else int(coarse_topk)

        self.class_mu: Dict[int, np.ndarray] = {}  # Normalized NCM centers for inference
        self.class_mu_raw: Dict[int, np.ndarray] = {}  # Raw (unnormalized) NCM centers for STAR
        self.class_count: Dict[int, int] = {}

        self.class_clusters: Dict[int, List[_Cluster]] = {}
        self.class_edges: Dict[int, Dict[int, Dict[int, int]]] = {}
        self.class_edge_reliability: Dict[int, Dict[int, Dict[int, float]]] = {}
        self.known_classes_before_task: Optional[int] = None
        
        # STAR: Store ORIGINAL clusters and NCM centers (before any transformation)
        # This allows re-alignment from scratch to avoid cumulative errors
        self.class_clusters_original: Dict[int, List[_Cluster]] = {}
        self.class_mu_original: Dict[int, np.ndarray] = {}  # Normalized
        self.class_mu_raw_original: Dict[int, np.ndarray] = {}  # Unnormalized

        # DEBUG: Store snapshots of clusters at different tasks for comparison
        # Format: {task_id: {cls: [centers_normalized, centers_raw]}}
        self.cluster_snapshots: Dict[int, Dict[int, tuple]] = {}
        
        # STAR: Frozen clusters for hard restore (Force Freeze Policy)
        # {cls: List[_Cluster]} - Stores deep copy of clusters
        self.frozen_clusters: Dict[int, List[_Cluster]] = {}

        self.buffers: Dict[int, List[np.ndarray]] = {}

        # ------------------------------------------------------------------ #
        # Inference acceleration cache (built lazily on first predict_topk call)
        # ------------------------------------------------------------------ #
        # Cache is invalidated whenever class centers / prototypes change (compress, alignment, etc.)
        self._predict_cache_dirty: bool = True
        self._predict_cache: Dict[str, object] = {}

        # ------------------------------------------------------------------ #
        # Inference profiling (optional; focused on predict_topk internals)
        # ------------------------------------------------------------------ #
        self.enable_inference_profiling = bool(enable_inference_profiling)
        self.profile_sync_cuda = bool(profile_sync_cuda)
        self._profile_stats: Dict[str, object] = {}
        self.reset_profile_stats()

        # ------------------------------------------------------------------ #
        # Dictionary coding switch (set by Agent D or externally)
        # ------------------------------------------------------------------ #
        self.use_dict_coding: bool = bool(use_dict_coding)
        self.dict_sparse_k: int = int(dict_sparse_k)
        self.dict_ridge_lambda: float = float(dict_ridge_lambda)
        self.drop_node_raw_after_dict_materialize: bool = bool(drop_node_raw_after_dict_materialize)

        # --- Dictionary growth thresholds (Agent F) ---
        self.theta_residual: float = float(theta_residual)
        self.max_growth_per_task: int = int(max_growth_per_task)

        # ------------------------------------------------------------------ #
        # Ablation switches (P0-4). Growth and additive edge-aware scoring are
        # retained for historical reproducibility only and are disabled by
        # default in all forward experiment settings.
        # ------------------------------------------------------------------ #
        self.use_dictionary_growth: bool = False
        self.use_lifecycle: bool = True
        self.use_protected_gate: bool = True
        self.use_edge_age_persistence: bool = True
        self.use_edge_aware_scoring: bool = False
        self.edge_score_gamma: float = 0.1
        self.edge_score_eta: float = 0.05
        self._last_edge_score_stats: Dict[str, float] = {}

        # ------------------------------------------------------------------ #
        # Direction 1: Reliability-calibrated raw-node fallback.
        # ------------------------------------------------------------------ #
        self.use_raw_fallback_gate: bool = bool(use_raw_fallback_gate)
        self.raw_fallback_per_class: int = max(0, int(raw_fallback_per_class))
        self.raw_fallback_select_by: str = str(raw_fallback_select_by)
        self.fallback_gate_type: str = str(fallback_gate_type)
        self.fallback_gate_margin_threshold: float = float(fallback_gate_margin_threshold)
        self.fallback_gate_residual_threshold: float = float(fallback_gate_residual_threshold)
        self.fallback_gate_random_rate: float = min(1.0, max(0.0, float(fallback_gate_random_rate)))
        self.fallback_memory_accounting: bool = bool(fallback_memory_accounting)
        self.fallback_calibration_samples_per_class: int = max(0, int(fallback_calibration_samples_per_class))
        self.fallback_gate_min_positives: int = max(0, int(fallback_gate_min_positives))
        self.fallback_gate_min_calibration_gain: float = max(0.0, float(fallback_gate_min_calibration_gain))
        self.fallback_gate_max_rate: float = min(1.0, max(0.0, float(fallback_gate_max_rate)))
        self.fallback_gate_tree_estimators: int = max(1, int(fallback_gate_tree_estimators))
        self.fallback_gate_tree_max_depth: int = max(1, int(fallback_gate_tree_max_depth))
        self.fallback_gate_tree_min_samples_leaf: int = max(1, int(fallback_gate_tree_min_samples_leaf))
        self.fallback_gate_threshold_strategy: str = str(fallback_gate_threshold_strategy).lower()
        self.fallback_gate_min_precision: float = min(1.0, max(0.0, float(fallback_gate_min_precision)))
        self.fallback_gate_min_selected: int = max(0, int(fallback_gate_min_selected))
        self.fallback_gate_min_net_count: int = max(0, int(fallback_gate_min_net_count))
        self.fallback_candidate_rule: str = str(fallback_candidate_rule)
        self.fallback_gate_budget_rates: Tuple[float, ...] = self._parse_fallback_budget_rates(
            fallback_gate_budget_rates
        )
        self.fallback_gate_neutral_weight: float = max(0.0, float(fallback_gate_neutral_weight))
        self.fallback_pair_table_mode: str = str(fallback_pair_table_mode)
        self.fallback_pair_table_smoothing: float = max(0.0, float(fallback_pair_table_smoothing))
        self.fallback_pair_table_lambda: float = float(fallback_pair_table_lambda)
        self.fallback_node_table_lambda: float = float(fallback_node_table_lambda)
        self.enable_prediction_trace: bool = bool(enable_prediction_trace)
        self.trace_split: str = str(trace_split)
        self._raw_fallback_rng = np.random.RandomState(int(fallback_random_seed))
        self.raw_fallback_cache: Dict[int, List[Dict[str, object]]] = {}
        self._raw_fallback_cache_version: int = 0
        self._raw_fallback_predict_cache_dirty: bool = True
        self._raw_fallback_predict_cache: Dict[str, object] = {}
        self._prediction_trace_records: List[Dict[str, object]] = []
        self._raw_fallback_gate_model: Dict[str, object] = {}
        self._raw_fallback_gate_model_dirty: bool = True
        self._raw_fallback_gate_fit_stats: Dict[str, float] = {}
        self._last_fallback_stats: Dict[str, float] = {}
        self._raw_fallback_eval_stats: Dict[str, float] = {}
        self.reset_raw_fallback_eval_stats(clear_trace=True)

        # ------------------------------------------------------------------ #
        # LifeTopoDict: shared dictionary atoms and metadata
        # ------------------------------------------------------------------ #
        # dict_atoms: atom matrix [M, d]; same as dict_atoms_hat (all rows L2-normalised).
        # Kept as a separate field for backward-compat with center_raw semantics.
        self.dict_atoms: Optional[np.ndarray] = None
        # dict_atoms_hat: row-L2-normalised atom matrix [M, d] — single source of truth.
        self.dict_atoms_hat: Optional[np.ndarray] = None
        # Per-atom lifecycle state: 'protected' | 'plastic' | 'inactive'
        self.atom_states: List[str] = []
        # Per-atom origin metadata: 'base' | 'grown'
        self.atom_origins: List[str] = []
        # Per-atom usage tracking: {atom_index: cumulative_usage}
        self.atom_usage: Dict[int, float] = {}
        # Per-atom EMA usage tracking (decays over time)
        self.atom_usage_ema: Dict[int, float] = {}          # EMA usage rate
        self.atom_usage_task: Dict[int, float] = {}          # per-task snapshot
        self._usage_ema_decay: float = 0.7                  # EMA decay factor

        # ------------------------------------------------------------------ #
        # Lifecycle state tracking (Agent E: E-1)
        # ------------------------------------------------------------------ #
        # Per-class per-node state: 'protected' / 'plastic' / 'inactive'
        self.node_states: Dict[int, List[str]] = {}

        # Per-atom old-class support strength (populated by _compute_old_support)
        self.old_support: Dict[int, float] = {}

        # Per-atom set of old task class indices that use this atom
        self.atom_old_tasks: Dict[int, Set[int]] = {}

        # Active/inactive atom mask (length = number of atoms)
        self.inactive_mask: Optional[np.ndarray] = None

        # Per-atom consecutive low-usage task counter
        self._low_usage_streak: Dict[int, int] = {}

        # Lifecycle thresholds
        self.lifecycle_theta_support = float(lifecycle_theta_support)
        self.lifecycle_theta_usage = float(lifecycle_theta_usage)
        self.lifecycle_T_inactive = int(lifecycle_T_inactive)
        self.lifecycle_min_alive_ratio = float(lifecycle_min_alive_ratio)
        self.lifecycle_protect_old_topk = bool(lifecycle_protect_old_topk)
        self.lifecycle_node_inactive_threshold = min(
            1.0, max(0.0, float(lifecycle_node_inactive_threshold))
        )
        self._last_lifecycle_summary: Dict[str, float] = {}
        self._last_predict_cache_filter_stats: Dict[str, float] = {}

        # Track whether dict_init has been called (for compress integration)
        self._dict_initialized: bool = False

        # --- GTE tracking (Agent F: F-5) ---
        # Stores per-task GTE records for diagnostics
        self._gte_records: List[Dict[str, float]] = []

    def set_task_boundary(self, known_classes_before_task: Optional[int]) -> None:
        """Record the class boundary used to distinguish old vs current classes."""
        self.known_classes_before_task = (
            None if known_classes_before_task is None else int(known_classes_before_task)
        )

    # ================================================================== #
    # Direction 1: raw-node fallback cache and prediction trace           #
    # ================================================================== #

    def reset_raw_fallback_eval_stats(self, clear_trace: bool = True) -> None:
        """Reset per-evaluation fallback/trace accumulators."""
        self._raw_fallback_eval_stats = {
            "samples": 0.0,
            "fallback_used": 0.0,
            "prediction_changed": 0.0,
            "compact_raw_disagree": 0.0,
            "compact_correct": 0.0,
            "final_correct": 0.0,
            "fallback_correct": 0.0,
            "compact_correct_when_used": 0.0,
            "fallback_correct_when_used": 0.0,
            "margin_sum": 0.0,
            "residual_sum": 0.0,
            "fallback_margin_sum": 0.0,
            "fallback_residual_sum": 0.0,
            "oracle_improvable": 0.0,
            "fallback_available": 0.0,
            "benefit_total": 0.0,
            "harm_total": 0.0,
            "neutral_total": 0.0,
            "benefit_selected": 0.0,
            "harm_selected": 0.0,
            "neutral_selected": 0.0,
        }
        self._last_fallback_stats = self._summarize_raw_fallback_eval_stats()
        if clear_trace:
            self._prediction_trace_records = []

    def get_prediction_trace(self) -> List[Dict[str, object]]:
        """Return the most recent eval prediction trace records."""
        return list(getattr(self, "_prediction_trace_records", []))

    def _fallback_cache_counts(self) -> Tuple[int, int]:
        cache = getattr(self, "raw_fallback_cache", {})
        node_count = int(sum(len(v) for v in cache.values()))
        class_count = int(sum(1 for v in cache.values() if len(v) > 0))
        return node_count, class_count

    def _fallback_candidate_score(self, entry: Dict[str, object]) -> float:
        select_by = str(getattr(self, "raw_fallback_select_by", "residual")).lower()
        residual = float(entry.get("residual", 0.0))
        count = float(entry.get("count", 0.0))
        if select_by == "random":
            return float(self._raw_fallback_rng.rand())
        if select_by == "count":
            return count
        if select_by in ("low_residual", "residual_low"):
            return -residual
        if select_by == "residual_margin":
            return residual * np.log1p(max(count, 0.0))
        if select_by == "risk":
            state = str(entry.get("node_state", "plastic"))
            state_bonus = 0.05 if state == "protected" else 0.0
            return residual + state_bonus + 0.01 * np.log1p(max(count, 0.0))
        return residual

    def _update_raw_fallback_cache(self, candidates: List[Dict[str, object]]) -> None:
        """Merge raw node snapshots into the audited fallback cache."""
        if self.raw_fallback_per_class <= 0:
            if self.raw_fallback_cache:
                self.raw_fallback_cache.clear()
                self._raw_fallback_cache_version += 1
                self._raw_fallback_predict_cache_dirty = True
            return

        if not candidates:
            return

        per_class: Dict[int, List[Dict[str, object]]] = {
            int(cls): list(entries) for cls, entries in self.raw_fallback_cache.items()
        }
        for entry in candidates:
            cls = int(entry["class_id"])
            per_class.setdefault(cls, []).append(entry)

        kept: Dict[int, List[Dict[str, object]]] = {}
        for cls, entries in per_class.items():
            latest_by_key: Dict[Tuple[int, int, int], Dict[str, object]] = {}
            for entry in entries:
                key = (
                    int(entry.get("class_id", cls)),
                    int(entry.get("node_id", -1)),
                    int(entry.get("source_task", -1)),
                )
                latest_by_key[key] = entry
            ranked = sorted(
                latest_by_key.values(),
                key=self._fallback_candidate_score,
                reverse=True,
            )
            kept[cls] = ranked[: self.raw_fallback_per_class]

        self.raw_fallback_cache = kept
        self._raw_fallback_cache_version += 1
        self._raw_fallback_predict_cache_dirty = True
        self._raw_fallback_predict_cache.clear()
        nodes, classes = self._fallback_cache_counts()
        logging.info(
            f"[RawFallback] cache updated: nodes={nodes}, classes={classes}, "
            f"per_class={self.raw_fallback_per_class}, select_by={self.raw_fallback_select_by}"
        )

    def _raw_fallback_storage_bytes(self) -> Dict[str, float]:
        """Return storage bytes for raw fallback vectors and metadata."""
        vector_bytes = 0.0
        metadata_bytes = 0.0
        for entries in getattr(self, "raw_fallback_cache", {}).values():
            for entry in entries:
                center_raw = entry.get("center_raw")
                if isinstance(center_raw, np.ndarray):
                    vector_bytes += float(center_raw.nbytes)
                metadata_bytes += 4  # class id
                metadata_bytes += 4  # node id
                metadata_bytes += 4  # count
                metadata_bytes += 8  # residual
                metadata_bytes += 4  # source task
                metadata_bytes += 4  # snapshot/version
                metadata_bytes += len(str(entry.get("node_state", "plastic")).encode("utf-8"))
        total_bytes = vector_bytes + metadata_bytes
        return {
            "raw_fallback_vector_bytes": vector_bytes,
            "raw_fallback_metadata_bytes": metadata_bytes,
            "raw_fallback_total_bytes": total_bytes,
        }

    def _ensure_raw_fallback_predict_cache(
        self,
        device: torch.device,
        query_dim: int,
        valid_classes: List[int],
    ) -> None:
        """Build cached tensors for raw fallback inference."""
        device_key = str(device)
        classes_key = tuple(int(c) for c in valid_classes)
        if (
            (not self._raw_fallback_predict_cache_dirty)
            and self._raw_fallback_predict_cache.get("device_key") == device_key
            and self._raw_fallback_predict_cache.get("query_dim") == int(query_dim)
            and self._raw_fallback_predict_cache.get("classes_key") == classes_key
            and self._raw_fallback_predict_cache.get("cache_version") == self._raw_fallback_cache_version
        ):
            return

        centers: List[np.ndarray] = []
        labels: List[int] = []
        class_indices: List[int] = []
        node_ids: List[int] = []
        counts: List[float] = []
        residuals: List[float] = []
        state_codes: List[int] = []
        state_map = {"inactive": 0, "plastic": 1, "protected": 2}

        class_to_index = {int(cls): idx for idx, cls in enumerate(valid_classes)}
        for cls in valid_classes:
            for entry in self.raw_fallback_cache.get(int(cls), []):
                center_raw = entry.get("center_raw")
                if not isinstance(center_raw, np.ndarray) or center_raw.shape[0] != query_dim:
                    continue
                centers.append(_normalize(center_raw.astype(np.float32, copy=False)))
                labels.append(int(cls))
                class_indices.append(int(class_to_index[int(cls)]))
                node_ids.append(int(entry.get("node_id", -1)))
                counts.append(float(entry.get("count", 0.0)))
                residuals.append(float(entry.get("residual", 0.0)))
                state_codes.append(state_map.get(str(entry.get("node_state", "plastic")), 1))

        if centers:
            centers_np = np.stack(centers, axis=0).astype(np.float32, copy=False)
            centers_t = torch.from_numpy(centers_np).to(device=device, dtype=torch.float32)
            labels_t = torch.tensor(labels, device=device, dtype=torch.long)
            class_indices_t = torch.tensor(class_indices, device=device, dtype=torch.long)
            node_ids_np = np.asarray(node_ids, dtype=np.int64)
            counts_np = np.asarray(counts, dtype=np.float32)
            residuals_np = np.asarray(residuals, dtype=np.float32)
            state_codes_np = np.asarray(state_codes, dtype=np.int64)
        else:
            centers_t = torch.empty((0, query_dim), device=device, dtype=torch.float32)
            labels_t = torch.empty((0,), device=device, dtype=torch.long)
            class_indices_t = torch.empty((0,), device=device, dtype=torch.long)
            node_ids_np = np.empty((0,), dtype=np.int64)
            counts_np = np.empty((0,), dtype=np.float32)
            residuals_np = np.empty((0,), dtype=np.float32)
            state_codes_np = np.empty((0,), dtype=np.int64)

        self._raw_fallback_predict_cache = {
            "device_key": device_key,
            "query_dim": int(query_dim),
            "classes_key": classes_key,
            "cache_version": self._raw_fallback_cache_version,
            "fallback_centers_t": centers_t,
            "fallback_labels_t": labels_t,
            "fallback_class_index_t": class_indices_t,
            "fallback_node_ids_np": node_ids_np,
            "fallback_counts_np": counts_np,
            "fallback_residuals_np": residuals_np,
            "fallback_state_codes_np": state_codes_np,
        }
        self._raw_fallback_predict_cache_dirty = False

    def _raw_fallback_gate_model_bytes(self) -> Dict[str, float]:
        """Return the storage footprint of the learned fallback gate."""
        model = getattr(self, "_raw_fallback_gate_model", {}) or {}
        total_bytes = 0.0
        for key in ("feature_mean", "feature_std", "weights", "compact_feature_indices", "raw_feature_indices"):
            value = model.get(key)
            if isinstance(value, np.ndarray):
                total_bytes += float(value.nbytes)
        for estimator in (
            model.get("estimator"),
            model.get("compact_estimator"),
            model.get("raw_estimator"),
        ):
            if estimator is None:
                continue
            try:
                import pickle

                total_bytes += float(len(pickle.dumps(estimator, protocol=4)))
            except Exception:
                total_bytes += 0.0
        if "bias" in model:
            total_bytes += 8.0
        if "threshold" in model:
            total_bytes += 8.0
        for table_key in ("class_pair_table", "class_pair_task_table", "node_pair_table"):
            table = model.get(table_key)
            if isinstance(table, dict):
                # Conservative deployable estimate: two int keys plus one float
                # utility value per entry. The Python dict itself is runtime
                # overhead and is counted under actual implementation memory.
                total_bytes += float(len(table) * (8 * 3))
        if "global_utility" in model:
            total_bytes += 8.0
        return {
            "raw_fallback_gate_model_bytes": total_bytes,
        }

    @staticmethod
    def _canonical_fallback_gate_type(gate_type: str) -> str:
        gate_type = str(gate_type).lower()
        if gate_type in ("gbdt", "gbdt_depth2", "gradient_boosting"):
            return "gbdt"
        if gate_type in ("rf", "random_forest", "rf_depth4"):
            return "random_forest"
        if gate_type in ("delta_ridge_pair", "utility_ridge_pair"):
            return "delta_ridge_pair"
        if gate_type in ("benefit_harm", "benefit_harm_logit", "utility_logistic"):
            return "benefit_harm_logistic"
        if gate_type in ("dual", "dual_logistic", "dual_calibrator", "dual_logistic_lcb"):
            return "dual_logistic"
        if gate_type in ("rule_candidate", "candidate_only", "candidate_only_rule"):
            return "candidate_rule"
        return gate_type

    @classmethod
    def _learned_fallback_gate_types(cls) -> Tuple[str, ...]:
        return (
            "ridge",
            "logistic",
            "gbdt",
            "random_forest",
            "candidate_rule",
            "delta_ridge",
            "delta_logistic",
            "benefit_harm_logistic",
            "dual_logistic",
            "utility_tree",
            "utility_gbdt",
            "utility_table",
            "delta_ridge_pair",
        )

    @staticmethod
    def _parse_fallback_budget_rates(value: object) -> Tuple[float, ...]:
        if value is None:
            return (0.002, 0.005, 0.01, 0.02, 0.04)
        if isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = str(value).replace(";", ",").split(",")
        rates = []
        for item in raw_values:
            try:
                rate = float(item)
            except (TypeError, ValueError):
                continue
            if rate > 1.0:
                rate = rate / 100.0
            if 0.0 < rate <= 1.0:
                rates.append(rate)
        if not rates:
            rates = [0.002, 0.005, 0.01, 0.02, 0.04]
        return tuple(sorted(set(float(r) for r in rates)))

    @staticmethod
    def _state_one_hot(state: str) -> np.ndarray:
        state = str(state).lower()
        return np.asarray(
            [
                1.0 if state == "inactive" else 0.0,
                1.0 if state == "plastic" else 0.0,
                1.0 if state == "protected" else 0.0,
            ],
            dtype=np.float32,
        )

    def _build_raw_fallback_gate_features(
        self,
        compact_margin: np.ndarray,
        selected_residual: np.ndarray,
        selected_distance: np.ndarray,
        fallback_distance: np.ndarray,
        selected_count: np.ndarray,
        fallback_residual: np.ndarray,
        fallback_count: np.ndarray,
        selected_state: np.ndarray,
        compact_top1_score: Optional[np.ndarray] = None,
        compact_top2_score: Optional[np.ndarray] = None,
        fallback_margin: Optional[np.ndarray] = None,
        raw_top1_score: Optional[np.ndarray] = None,
        raw_top2_score: Optional[np.ndarray] = None,
        selected_atom_weighted_inactive_ratio: Optional[np.ndarray] = None,
        selected_atom_old_support_mean: Optional[np.ndarray] = None,
        selected_atom_weighted_old_support: Optional[np.ndarray] = None,
        selected_atom_usage_ema_mean: Optional[np.ndarray] = None,
        selected_atom_weighted_usage_ema: Optional[np.ndarray] = None,
        selected_atom_entropy: Optional[np.ndarray] = None,
        compact_raw_disagree: Optional[np.ndarray] = None,
        compact_rank_of_raw_top1: Optional[np.ndarray] = None,
        raw_rank_of_compact_top1: Optional[np.ndarray] = None,
        compact_score_on_raw_top1: Optional[np.ndarray] = None,
        raw_score_on_compact_top1: Optional[np.ndarray] = None,
        score_cross_gap: Optional[np.ndarray] = None,
        selected_class_age: Optional[np.ndarray] = None,
        raw_class_age: Optional[np.ndarray] = None,
        task_relation_code: Optional[np.ndarray] = None,
        fallback_available: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Construct feature rows used by the learned fallback gate."""
        compact_margin = np.asarray(compact_margin, dtype=np.float32).reshape(-1)
        n = int(compact_margin.shape[0])

        def _optional_array(values, default=0.0):
            if values is None:
                arr = np.full(n, float(default), dtype=np.float32)
            else:
                arr = np.asarray(values, dtype=np.float32).reshape(-1)
                if arr.shape[0] != n:
                    arr = np.resize(arr, n).astype(np.float32, copy=False)
            return np.where(np.isfinite(arr), arr, float(default)).astype(np.float32, copy=False)

        selected_residual = np.asarray(selected_residual, dtype=np.float32).reshape(-1)
        selected_distance = np.asarray(selected_distance, dtype=np.float32).reshape(-1)
        fallback_distance = np.asarray(fallback_distance, dtype=np.float32).reshape(-1)
        selected_count = np.asarray(selected_count, dtype=np.float32).reshape(-1)
        fallback_residual = np.asarray(fallback_residual, dtype=np.float32).reshape(-1)
        fallback_count = np.asarray(fallback_count, dtype=np.float32).reshape(-1)
        selected_state = np.asarray(selected_state).reshape(-1)

        compact_margin = np.where(np.isfinite(compact_margin), compact_margin, 0.0)
        selected_residual = np.where(np.isfinite(selected_residual), selected_residual, 0.0)
        selected_distance = np.where(np.isfinite(selected_distance), selected_distance, 0.0)
        fallback_distance = np.where(np.isfinite(fallback_distance), fallback_distance, selected_distance)
        selected_count = np.where(np.isfinite(selected_count), selected_count, 0.0)
        fallback_residual = np.where(np.isfinite(fallback_residual), fallback_residual, 0.0)
        fallback_count = np.where(np.isfinite(fallback_count), fallback_count, 0.0)
        compact_top1_score = _optional_array(compact_top1_score, 0.0)
        compact_top2_score = _optional_array(compact_top2_score, 0.0)
        fallback_margin = _optional_array(fallback_margin, 0.0)
        raw_top1_score = _optional_array(raw_top1_score, 0.0)
        raw_top2_score = _optional_array(raw_top2_score, 0.0)
        selected_atom_weighted_inactive_ratio = _optional_array(
            selected_atom_weighted_inactive_ratio, 0.0
        )
        selected_atom_old_support_mean = _optional_array(selected_atom_old_support_mean, 0.0)
        selected_atom_weighted_old_support = _optional_array(
            selected_atom_weighted_old_support, 0.0
        )
        selected_atom_usage_ema_mean = _optional_array(selected_atom_usage_ema_mean, 0.0)
        selected_atom_weighted_usage_ema = _optional_array(
            selected_atom_weighted_usage_ema, 0.0
        )
        selected_atom_entropy = _optional_array(selected_atom_entropy, 0.0)
        compact_raw_disagree = _optional_array(compact_raw_disagree, 0.0)
        compact_rank_of_raw_top1 = _optional_array(compact_rank_of_raw_top1, 999.0)
        raw_rank_of_compact_top1 = _optional_array(raw_rank_of_compact_top1, 999.0)
        compact_score_on_raw_top1 = _optional_array(compact_score_on_raw_top1, 0.0)
        raw_score_on_compact_top1 = _optional_array(raw_score_on_compact_top1, 0.0)
        score_cross_gap = _optional_array(score_cross_gap, 0.0)
        selected_class_age = _optional_array(selected_class_age, 0.0)
        raw_class_age = _optional_array(raw_class_age, 0.0)
        task_relation_code = _optional_array(task_relation_code, 0.0)
        fallback_available = _optional_array(fallback_available, 0.0)

        distance_gap = selected_distance - fallback_distance
        compact_margin_safe = np.maximum(compact_margin, 1e-6)
        fallback_margin_ratio = np.clip(fallback_margin / compact_margin_safe, -50.0, 50.0)
        distance_gap_over_margin = np.clip(distance_gap / compact_margin_safe, -50.0, 50.0)
        compact_fallback_score_gap = compact_top1_score - fallback_distance
        residual_gap = selected_residual - fallback_residual
        features = [
            compact_margin,
            selected_residual,
            selected_distance,
            fallback_distance,
            distance_gap,
            np.log1p(np.maximum(selected_count, 0.0)),
            fallback_residual,
            np.log1p(np.maximum(fallback_count, 0.0)),
            np.log1p(np.maximum(fallback_count, 0.0)) - np.log1p(np.maximum(selected_count, 0.0)),
            compact_top1_score,
            compact_top2_score,
            fallback_margin,
            raw_top1_score,
            raw_top2_score,
            fallback_margin_ratio,
            distance_gap_over_margin,
            compact_fallback_score_gap,
            residual_gap,
            (fallback_distance < selected_distance).astype(np.float32),
            selected_atom_weighted_inactive_ratio,
            selected_atom_old_support_mean,
            selected_atom_weighted_old_support,
            selected_atom_usage_ema_mean,
            selected_atom_weighted_usage_ema,
            selected_atom_entropy,
            compact_raw_disagree,
            np.clip(compact_rank_of_raw_top1, 0.0, 999.0),
            np.clip(raw_rank_of_compact_top1, 0.0, 999.0),
            compact_score_on_raw_top1,
            raw_score_on_compact_top1,
            score_cross_gap,
            selected_class_age,
            raw_class_age,
            task_relation_code,
            fallback_available,
        ]
        state_one_hot = np.stack([self._state_one_hot(s) for s in selected_state], axis=0)
        features.append(state_one_hot[:, 0])
        features.append(state_one_hot[:, 1])
        features.append(state_one_hot[:, 2])
        feature_names = [
            "compact_margin",
            "selected_residual",
            "selected_distance",
            "fallback_distance",
            "distance_gap",
            "selected_count_log",
            "fallback_residual",
            "fallback_count_log",
            "count_log_gap",
            "compact_top1_score",
            "compact_top2_score",
            "fallback_margin",
            "raw_top1_score",
            "raw_top2_score",
            "fallback_margin_ratio",
            "distance_gap_over_margin",
            "compact_fallback_score_gap",
            "residual_gap",
            "fallback_better_distance",
            "selected_atom_weighted_inactive_ratio",
            "selected_atom_old_support_mean",
            "selected_atom_weighted_old_support",
            "selected_atom_usage_ema_mean",
            "selected_atom_weighted_usage_ema",
            "selected_atom_entropy",
            "compact_raw_disagree",
            "compact_rank_of_raw_top1",
            "raw_rank_of_compact_top1",
            "compact_score_on_raw_top1",
            "raw_score_on_compact_top1",
            "score_cross_gap",
            "selected_class_age",
            "raw_class_age",
            "task_relation_code",
            "fallback_available",
            "selected_state_inactive",
            "selected_state_plastic",
            "selected_state_protected",
        ]
        return np.stack(features, axis=1).astype(np.float32, copy=False), feature_names

    def _fallback_candidate_rule_name(self) -> str:
        rule = str(getattr(self, "fallback_candidate_rule", "none")).lower()
        if rule in ("", "all"):
            return "none"
        return rule

    @staticmethod
    def _nanquantile(values: np.ndarray, q: float, default: float = 0.0) -> float:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float(default)
        return float(np.quantile(arr, q))

    def _fit_fallback_candidate_params(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        params = {
            "compact_margin_q30": self._nanquantile(data.get("compact_margin", []), 0.30),
            "raw_margin_q70": self._nanquantile(data.get("fallback_margin", []), 0.70),
            "selected_residual_q50": self._nanquantile(data.get("selected_residual", []), 0.50),
            "score_cross_gap_q70": self._nanquantile(data.get("score_cross_gap", []), 0.70),
        }
        return params

    def _build_raw_fallback_candidate_mask(
        self,
        data: Dict[str, np.ndarray],
        params: Optional[Dict[str, float]] = None,
        rule: Optional[str] = None,
    ) -> np.ndarray:
        """Build a deployable candidate mask before the learned utility score.

        The mask intentionally uses only prediction-side quantities. Labels are
        never touched here; label-derived benefit/harm is used only after this
        mask for calibration scoring.
        """
        rule = self._fallback_candidate_rule_name() if rule is None else str(rule).lower()
        fallback_available = np.asarray(data.get("fallback_available", True), dtype=bool).reshape(-1)
        n = int(fallback_available.shape[0])
        compact_top1 = np.asarray(data.get("compact_top1", np.full(n, -1)), dtype=np.int64).reshape(-1)
        fallback_top1 = np.asarray(data.get("fallback_top1", np.full(n, -2)), dtype=np.int64).reshape(-1)
        disagree = np.logical_and(fallback_available, compact_top1 != fallback_top1)
        if rule in ("none", "all"):
            return fallback_available.copy()
        if rule in ("disagree", "compact_raw_disagree"):
            return disagree

        params = params or self._fit_fallback_candidate_params(data)
        compact_margin = np.asarray(data.get("compact_margin", np.zeros(n)), dtype=np.float32).reshape(-1)
        raw_margin = np.asarray(data.get("fallback_margin", np.zeros(n)), dtype=np.float32).reshape(-1)
        compact_rank_of_raw = np.asarray(
            data.get("compact_rank_of_raw_top1", np.full(n, 999)),
            dtype=np.float32,
        ).reshape(-1)
        raw_rank_of_compact = np.asarray(
            data.get("raw_rank_of_compact_top1", np.full(n, 999)),
            dtype=np.float32,
        ).reshape(-1)
        selected_residual = np.asarray(
            data.get("selected_residual", np.zeros(n)),
            dtype=np.float32,
        ).reshape(-1)
        score_cross_gap = np.asarray(
            data.get("score_cross_gap", np.zeros(n)),
            dtype=np.float32,
        ).reshape(-1)

        margin_raw = (
            disagree
            & (compact_margin <= float(params.get("compact_margin_q30", 0.0)))
            & (raw_margin >= float(params.get("raw_margin_q70", 0.0)))
        )
        if rule in ("margin_raw", "basic", "basic_margin_raw"):
            return margin_raw

        enhanced = (
            margin_raw
            & (compact_rank_of_raw <= 3.0)
            & (raw_rank_of_compact > 1.0)
            & (selected_residual >= float(params.get("selected_residual_q50", 0.0)))
            & (score_cross_gap >= float(params.get("score_cross_gap_q70", 0.0)))
        )
        if rule in ("enhanced", "strict"):
            return enhanced
        return disagree

    def _select_raw_fallback_gate_threshold_budgeted(
        self,
        scores: np.ndarray,
        compact_correct: np.ndarray,
        fallback_correct: np.ndarray,
        candidate_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        compact_correct = np.asarray(compact_correct, dtype=bool).reshape(-1)
        fallback_correct = np.asarray(fallback_correct, dtype=bool).reshape(-1)
        n = int(scores.shape[0])
        if candidate_mask is None:
            candidate_mask = np.ones(n, dtype=bool)
        else:
            candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)

        finite_candidate = candidate_mask & np.isfinite(scores)
        benefit = (~compact_correct) & fallback_correct
        harm = compact_correct & (~fallback_correct)
        neutral = ~(benefit | harm)
        compact_acc = float(np.mean(compact_correct)) if n > 0 else 0.0

        best = {
            "threshold": float("inf"),
            "mask": np.zeros(n, dtype=bool),
            "net_gain_count": 0.0,
            "net_gain_rate": 0.0,
            "fallback_rate": 0.0,
            "benefit_selected": 0.0,
            "harm_selected": 0.0,
            "neutral_selected": 0.0,
            "benefit_recall": 0.0,
            "utility_precision": 0.0,
            "final_accuracy": compact_acc,
            "selected_budget_rate": 0.0,
            "net_gain_lcb": 0.0,
            "net_gain_std": 0.0,
        }
        candidate_indices = np.where(finite_candidate)[0]
        if candidate_indices.size == 0 or n == 0:
            return best["threshold"], best["mask"], {k: v for k, v in best.items() if k != "mask"}

        order = candidate_indices[np.argsort(-scores[candidate_indices])]
        max_rate = float(getattr(self, "fallback_gate_max_rate", 1.0))
        budgets = [b for b in getattr(self, "fallback_gate_budget_rates", ()) if b <= max_rate + 1e-12]
        if not budgets and max_rate > 0.0:
            budgets = [max_rate]

        for budget_rate in budgets:
            k = int(np.floor(float(budget_rate) * n + 1e-12))
            if k <= 0:
                k = 1
            k = min(k, int(order.size))
            if k <= 0:
                continue
            selected = order[:k]
            threshold = float(scores[selected[-1]])
            mask = finite_candidate & (scores >= threshold)
            rate = float(np.mean(mask))
            if rate > max_rate + 1e-12:
                continue
            b = float(np.sum(mask & benefit))
            h = float(np.sum(mask & harm))
            z = float(np.sum(mask & neutral))
            net = b - h
            net_rate = net / float(n)
            final_acc = compact_acc + net_rate
            precision = b / max(1.0, b + h)
            recall = b / max(1.0, float(np.sum(benefit)))
            selected_delta = np.zeros(int(np.sum(mask)), dtype=np.float64)
            if selected_delta.size > 0:
                selected_indices = np.where(mask)[0]
                selected_delta = np.where(
                    benefit[selected_indices],
                    1.0,
                    np.where(harm[selected_indices], -1.0, 0.0),
                )
            if selected_delta.size > 1:
                net_std = float(np.std(selected_delta, ddof=1) * np.sqrt(selected_delta.size) / float(n))
            else:
                net_std = 0.0
            net_lcb = net_rate - net_std
            strategy = str(getattr(self, "fallback_gate_threshold_strategy", "max_net")).lower()
            min_precision = float(getattr(self, "fallback_gate_min_precision", 0.0))
            min_selected = int(getattr(self, "fallback_gate_min_selected", 0))
            min_net_count = float(getattr(self, "fallback_gate_min_net_count", 0))
            selected_count = float(np.sum(mask))
            eligible = net_rate > 0.0
            if strategy in ("lcb", "lower_confidence_bound", "conservative_lcb"):
                eligible = net_lcb > 0.0
            eligible = (
                eligible
                and precision >= min_precision
                and selected_count >= float(min_selected)
                and net >= min_net_count
            )
            if not eligible:
                continue
            best_metric = float(best.get("net_gain_lcb", 0.0)) if strategy in (
                "lcb",
                "lower_confidence_bound",
                "conservative_lcb",
            ) else float(best["net_gain_rate"])
            current_metric = net_lcb if strategy in (
                "lcb",
                "lower_confidence_bound",
                "conservative_lcb",
            ) else net_rate
            better = False
            if current_metric > best_metric + 1e-12:
                better = True
            elif abs(current_metric - best_metric) <= 1e-12:
                if precision > float(best["utility_precision"]) + 1e-12:
                    better = True
                elif abs(precision - float(best["utility_precision"])) <= 1e-12 and rate < float(best["fallback_rate"]):
                    better = True
            if better:
                best = {
                    "threshold": threshold,
                    "mask": mask,
                    "net_gain_count": net,
                    "net_gain_rate": net_rate,
                    "fallback_rate": rate,
                    "benefit_selected": b,
                    "harm_selected": h,
                    "neutral_selected": z,
                    "benefit_recall": recall,
                    "utility_precision": precision,
                    "final_accuracy": final_acc,
                    "selected_budget_rate": float(budget_rate),
                    "net_gain_lcb": net_lcb,
                    "net_gain_std": net_std,
                }

        return best["threshold"], best["mask"], {k: v for k, v in best.items() if k != "mask"}

    def _fit_fallback_utility_tables(
        self,
        compact_top1: np.ndarray,
        fallback_top1: np.ndarray,
        selected_node: np.ndarray,
        task_relation_code: np.ndarray,
        compact_correct: np.ndarray,
        fallback_correct: np.ndarray,
    ) -> Dict[str, object]:
        smoothing = float(getattr(self, "fallback_pair_table_smoothing", 10.0))
        benefit = (~compact_correct) & fallback_correct
        harm = compact_correct & (~fallback_correct)

        def _accumulate(keys):
            counts: Dict[str, List[float]] = {}
            for key, b, h in zip(keys, benefit, harm):
                if not b and not h:
                    continue
                key = str(key)
                if key not in counts:
                    counts[key] = [0.0, 0.0]
                if b:
                    counts[key][0] += 1.0
                elif h:
                    counts[key][1] += 1.0
            return {
                key: float((bh[0] - bh[1]) / (bh[0] + bh[1] + smoothing))
                for key, bh in counts.items()
            }

        compact_top1 = np.asarray(compact_top1, dtype=np.int64).reshape(-1)
        fallback_top1 = np.asarray(fallback_top1, dtype=np.int64).reshape(-1)
        selected_node = np.asarray(selected_node, dtype=np.int64).reshape(-1)
        task_relation_code = np.asarray(task_relation_code, dtype=np.int64).reshape(-1)
        class_pair_keys = [f"{int(c)}|{int(r)}" for c, r in zip(compact_top1, fallback_top1)]
        class_pair_task_keys = [
            f"{int(c)}|{int(r)}|{int(t)}"
            for c, r, t in zip(compact_top1, fallback_top1, task_relation_code)
        ]
        node_pair_keys = [
            f"{int(c)}|{int(n)}|{int(r)}"
            for c, n, r in zip(compact_top1, selected_node, fallback_top1)
        ]
        b_total = float(np.sum(benefit))
        h_total = float(np.sum(harm))
        global_utility = float((b_total - h_total) / (b_total + h_total + smoothing))
        return {
            "class_pair_table": _accumulate(class_pair_keys),
            "class_pair_task_table": _accumulate(class_pair_task_keys),
            "node_pair_table": _accumulate(node_pair_keys),
            "global_utility": global_utility,
        }

    def _lookup_fallback_utility_tables(
        self,
        model: Dict[str, object],
        compact_top1: np.ndarray,
        fallback_top1: np.ndarray,
        selected_node: np.ndarray,
        task_relation_code: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        compact_top1 = np.asarray(compact_top1, dtype=np.int64).reshape(-1)
        fallback_top1 = np.asarray(fallback_top1, dtype=np.int64).reshape(-1)
        selected_node = np.asarray(selected_node, dtype=np.int64).reshape(-1)
        task_relation_code = np.asarray(task_relation_code, dtype=np.int64).reshape(-1)
        n = int(compact_top1.shape[0])
        class_pair = model.get("class_pair_table", {}) if isinstance(model, dict) else {}
        class_pair_task = model.get("class_pair_task_table", {}) if isinstance(model, dict) else {}
        node_pair = model.get("node_pair_table", {}) if isinstance(model, dict) else {}
        global_utility = float(model.get("global_utility", 0.0)) if isinstance(model, dict) else 0.0
        pair_scores = np.full(n, global_utility, dtype=np.float32)
        node_scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            cp_key = f"{int(compact_top1[i])}|{int(fallback_top1[i])}"
            cpt_key = f"{int(compact_top1[i])}|{int(fallback_top1[i])}|{int(task_relation_code[i])}"
            np_key = f"{int(compact_top1[i])}|{int(selected_node[i])}|{int(fallback_top1[i])}"
            if isinstance(class_pair, dict) and cp_key in class_pair:
                pair_scores[i] = float(class_pair[cp_key])
            if isinstance(class_pair_task, dict) and cpt_key in class_pair_task:
                pair_scores[i] = float(class_pair_task[cpt_key])
            if isinstance(node_pair, dict) and np_key in node_pair:
                node_scores[i] = float(node_pair[np_key])
        return pair_scores, node_scores

    def _select_raw_fallback_gate_threshold(
        self,
        scores: np.ndarray,
        compact_correct: np.ndarray,
        fallback_correct: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        compact_correct = np.asarray(compact_correct, dtype=bool).reshape(-1)
        fallback_correct = np.asarray(fallback_correct, dtype=bool).reshape(-1)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            return float("inf"), np.zeros(scores.shape[0], dtype=bool)

        candidate_scores = np.unique(finite_scores)
        candidate_thresholds = np.concatenate(
            [
                np.asarray([candidate_scores.min() - 1e-6], dtype=np.float64),
                candidate_scores,
                np.asarray([candidate_scores.max() + 1e-6], dtype=np.float64),
            ]
        )
        max_rate = float(getattr(self, "fallback_gate_max_rate", 1.0))
        best_acc = -1.0
        best_rate = float("inf")
        best_threshold = float(candidate_thresholds[-1])
        best_mask = np.zeros(scores.shape[0], dtype=bool)
        for thr in candidate_thresholds:
            gate_mask = np.isfinite(scores) & (scores >= thr)
            rate = float(np.mean(gate_mask)) if gate_mask.size > 0 else 0.0
            if rate > max_rate + 1e-12:
                continue
            final_correct = np.where(gate_mask, fallback_correct, compact_correct)
            acc = float(np.mean(final_correct)) if final_correct.size > 0 else 0.0
            if (acc > best_acc + 1e-12) or (abs(acc - best_acc) <= 1e-12 and rate < best_rate):
                best_acc = acc
                best_rate = rate
                best_threshold = float(thr)
                best_mask = gate_mask
        return best_threshold, best_mask

    def fit_raw_fallback_gate_from_trace(
        self,
        trace_records: List[Dict[str, object]],
    ) -> Dict[str, float]:
        """Fit a deployable raw-fallback gate from calibration trace records.

        Legacy gate types (``ridge/logistic/gbdt/random_forest``) keep the old
        oracle-positive classifier objective for controlled comparison.
        Direction-1 utility gates instead learn or rank by
        ``delta = 1[raw correct] - 1[compact correct]`` and choose a threshold
        from a fixed validation fallback-budget set.
        """
        gate_type = self._canonical_fallback_gate_type(
            str(getattr(self, "fallback_gate_type", "rule")).lower()
        )
        if gate_type not in self._learned_fallback_gate_types():
            self._raw_fallback_gate_model = {}
            self._raw_fallback_gate_model_dirty = True
            self._raw_fallback_gate_fit_stats = {}
            return {}

        if not trace_records:
            self._raw_fallback_gate_model = {}
            self._raw_fallback_gate_model_dirty = True
            self._raw_fallback_gate_fit_stats = {"calibration_samples": 0.0}
            return self._raw_fallback_gate_fit_stats

        filtered = []
        for rec in trace_records:
            if not bool(rec.get("fallback_available", False)):
                continue
            if "compact_correct" not in rec or "fallback_correct" not in rec:
                continue
            filtered.append(rec)

        if not filtered:
            self._raw_fallback_gate_model = {}
            self._raw_fallback_gate_model_dirty = True
            self._raw_fallback_gate_fit_stats = {"calibration_samples": 0.0}
            return self._raw_fallback_gate_fit_stats

        def _array(name: str, default=0.0, dtype=np.float32):
            return np.asarray([rec.get(name, default) for rec in filtered], dtype=dtype)

        compact_margin = np.asarray([rec.get("compact_margin", 0.0) for rec in filtered], dtype=np.float32)
        compact_top1_score = np.asarray([rec.get("compact_top1_score", 0.0) for rec in filtered], dtype=np.float32)
        compact_top2_score = np.asarray([rec.get("compact_top2_score", 0.0) for rec in filtered], dtype=np.float32)
        compact_top1 = _array("compact_top1", -1, dtype=np.int64)
        fallback_top1 = _array("fallback_top1", -1, dtype=np.int64)
        selected_residual = np.asarray([rec.get("selected_node_residual", 0.0) for rec in filtered], dtype=np.float32)
        selected_distance = np.asarray([rec.get("nearest_compact_distance", 0.0) for rec in filtered], dtype=np.float32)
        fallback_distance = np.asarray([rec.get("fallback_distance", np.nan) for rec in filtered], dtype=np.float32)
        fallback_margin = np.asarray([rec.get("fallback_margin", 0.0) for rec in filtered], dtype=np.float32)
        raw_top1_score = _array("raw_top1_score", 0.0, dtype=np.float32)
        raw_top2_score = _array("raw_top2_score", 0.0, dtype=np.float32)
        selected_count = np.asarray(
            [rec.get("nearest_compact_count", rec.get("selected_node_count", 0.0)) for rec in filtered],
            dtype=np.float32,
        )
        fallback_residual = np.asarray([rec.get("fallback_residual", 0.0) for rec in filtered], dtype=np.float32)
        fallback_count = np.asarray([rec.get("fallback_count", 0.0) for rec in filtered], dtype=np.float32)
        selected_state = np.asarray([rec.get("selected_node_state", "plastic") for rec in filtered], dtype=object)
        selected_atom_weighted_inactive_ratio = np.asarray(
            [rec.get("selected_atom_weighted_inactive_ratio", 0.0) for rec in filtered],
            dtype=np.float32,
        )
        selected_atom_old_support_mean = np.asarray(
            [rec.get("selected_atom_old_support_mean", 0.0) for rec in filtered],
            dtype=np.float32,
        )
        selected_atom_weighted_old_support = np.asarray(
            [rec.get("selected_atom_weighted_old_support", 0.0) for rec in filtered],
            dtype=np.float32,
        )
        selected_atom_usage_ema_mean = np.asarray(
            [rec.get("selected_atom_usage_ema_mean", 0.0) for rec in filtered],
            dtype=np.float32,
        )
        selected_atom_weighted_usage_ema = np.asarray(
            [rec.get("selected_atom_weighted_usage_ema", 0.0) for rec in filtered],
            dtype=np.float32,
        )
        selected_atom_entropy = np.asarray(
            [rec.get("selected_atom_entropy", 0.0) for rec in filtered],
            dtype=np.float32,
        )
        fallback_available = np.asarray([bool(rec.get("fallback_available", False)) for rec in filtered], dtype=bool)
        selected_node = _array("nearest_compact_node", -1, dtype=np.int64)
        compact_raw_disagree = np.asarray(compact_top1 != fallback_top1, dtype=np.float32)
        compact_rank_of_raw_top1 = _array("compact_rank_of_raw_top1", 999.0, dtype=np.float32)
        raw_rank_of_compact_top1 = _array("raw_rank_of_compact_top1", 999.0, dtype=np.float32)
        compact_score_on_raw_top1 = _array("compact_score_on_raw_top1", 0.0, dtype=np.float32)
        raw_score_on_compact_top1 = _array("raw_score_on_compact_top1", 0.0, dtype=np.float32)
        score_cross_gap = _array("score_cross_gap", 0.0, dtype=np.float32)
        selected_class_age = _array("selected_class_age", 0.0, dtype=np.float32)
        raw_class_age = _array("raw_class_age", 0.0, dtype=np.float32)
        task_relation_code = _array("task_relation_code", 0.0, dtype=np.float32)
        compact_correct = np.asarray([bool(rec.get("compact_correct", False)) for rec in filtered], dtype=bool)
        fallback_correct = np.asarray([bool(rec.get("fallback_correct", False)) for rec in filtered], dtype=bool)
        benefit = np.logical_and(~compact_correct, fallback_correct)
        harm = np.logical_and(compact_correct, ~fallback_correct)
        both_correct = np.logical_and(compact_correct, fallback_correct)
        both_wrong = np.logical_and(~compact_correct, ~fallback_correct)
        neutral = np.logical_or(both_correct, both_wrong)
        delta_target = fallback_correct.astype(np.float32) - compact_correct.astype(np.float32)
        gate_target = benefit.astype(np.float32)

        X, feature_names = self._build_raw_fallback_gate_features(
            compact_margin=compact_margin,
            selected_residual=selected_residual,
            selected_distance=selected_distance,
            fallback_distance=fallback_distance,
            selected_count=selected_count,
            fallback_residual=fallback_residual,
            fallback_count=fallback_count,
            selected_state=selected_state,
            compact_top1_score=compact_top1_score,
            compact_top2_score=compact_top2_score,
            fallback_margin=fallback_margin,
            raw_top1_score=raw_top1_score,
            raw_top2_score=raw_top2_score,
            selected_atom_weighted_inactive_ratio=selected_atom_weighted_inactive_ratio,
            selected_atom_old_support_mean=selected_atom_old_support_mean,
            selected_atom_weighted_old_support=selected_atom_weighted_old_support,
            selected_atom_usage_ema_mean=selected_atom_usage_ema_mean,
            selected_atom_weighted_usage_ema=selected_atom_weighted_usage_ema,
            selected_atom_entropy=selected_atom_entropy,
            compact_raw_disagree=compact_raw_disagree,
            compact_rank_of_raw_top1=compact_rank_of_raw_top1,
            raw_rank_of_compact_top1=raw_rank_of_compact_top1,
            compact_score_on_raw_top1=compact_score_on_raw_top1,
            raw_score_on_compact_top1=raw_score_on_compact_top1,
            score_cross_gap=score_cross_gap,
            selected_class_age=selected_class_age,
            raw_class_age=raw_class_age,
            task_relation_code=task_relation_code,
            fallback_available=fallback_available.astype(np.float32),
        )
        n_samples, n_features = X.shape
        if n_samples == 0 or n_features == 0:
            self._raw_fallback_gate_model = {}
            self._raw_fallback_gate_model_dirty = True
            self._raw_fallback_gate_fit_stats = {"calibration_samples": 0.0}
            return self._raw_fallback_gate_fit_stats

        feature_mean = X.mean(axis=0, dtype=np.float64)
        feature_std = X.std(axis=0, dtype=np.float64)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        Xn = (X - feature_mean) / feature_std

        data_for_candidates = {
            "fallback_available": fallback_available,
            "compact_top1": compact_top1,
            "fallback_top1": fallback_top1,
            "compact_margin": compact_margin,
            "fallback_margin": fallback_margin,
            "selected_residual": selected_residual,
            "compact_rank_of_raw_top1": compact_rank_of_raw_top1,
            "raw_rank_of_compact_top1": raw_rank_of_compact_top1,
            "score_cross_gap": score_cross_gap,
        }
        candidate_params = self._fit_fallback_candidate_params(data_for_candidates)
        utility_gate_types = {
            "candidate_rule",
            "delta_ridge",
            "delta_logistic",
            "benefit_harm_logistic",
            "dual_logistic",
            "utility_tree",
            "utility_gbdt",
            "utility_table",
            "delta_ridge_pair",
        }
        candidate_rule = self._fallback_candidate_rule_name()
        if gate_type in utility_gate_types and candidate_rule == "none":
            candidate_rule = "disagree"
        candidate_mask = self._build_raw_fallback_candidate_mask(
            data_for_candidates,
            params=candidate_params,
            rule=candidate_rule,
        )

        pos_count = float(np.sum(gate_target > 0.5))
        neg_count = float(n_samples - pos_count)
        weights = np.zeros(n_features, dtype=np.float64)
        bias = 0.0
        estimator = None
        compact_estimator = None
        raw_estimator = None
        compact_feature_indices: List[int] = []
        raw_feature_indices: List[int] = []
        pos_weight = 1.0
        pair_tables: Dict[str, object] = {}
        pair_mode = str(getattr(self, "fallback_pair_table_mode", "none")).lower()
        if gate_type == "delta_ridge_pair" and pair_mode in ("", "none"):
            pair_mode = "full"

        if gate_type not in utility_gate_types:
            if pos_count <= 0.0:
                threshold = float("inf")
                scores = np.full(n_samples, -np.inf, dtype=np.float64)
            elif neg_count <= 0.0:
                threshold = float("-inf")
                scores = np.full(n_samples, np.inf, dtype=np.float64)
            else:
                y = gate_target.astype(np.float64, copy=False)
                pos_weight = min(10.0, max(1.0, float(np.sqrt(neg_count / max(1.0, pos_count)))))
                sample_weight = np.ones(n_samples, dtype=np.float64)
                sample_weight[y > 0.5] = pos_weight
                if gate_type in ("ridge", "logistic"):
                    Xaug = np.concatenate(
                        [
                            np.ones((n_samples, 1), dtype=np.float64),
                            Xn.astype(np.float64, copy=False),
                        ],
                        axis=1,
                    )
                    sqrt_w = np.sqrt(sample_weight)[:, None]
                    Xw = Xaug * sqrt_w
                    yw = y * np.sqrt(sample_weight)
                    reg_lambda = 1.0
                    gram = Xw.T @ Xw
                    reg = np.eye(gram.shape[0], dtype=np.float64) * reg_lambda
                    reg[0, 0] = 0.0
                    try:
                        coef = np.linalg.solve(gram + reg, Xw.T @ yw)
                    except np.linalg.LinAlgError:
                        coef = np.linalg.lstsq(gram + reg, Xw.T @ yw, rcond=None)[0]
                    bias = float(coef[0])
                    weights = coef[1:]
                    scores = Xaug @ coef
                elif gate_type == "gbdt":
                    try:
                        from sklearn.ensemble import GradientBoostingClassifier

                        estimator = GradientBoostingClassifier(
                            n_estimators=int(getattr(self, "fallback_gate_tree_estimators", 160)),
                            learning_rate=0.04,
                            max_depth=int(getattr(self, "fallback_gate_tree_max_depth", 2)),
                            min_samples_leaf=int(
                                getattr(self, "fallback_gate_tree_min_samples_leaf", 12)
                            ),
                            random_state=0,
                        )
                        estimator.fit(Xn.astype(np.float32, copy=False), y.astype(np.int64), sample_weight=sample_weight)
                        scores = estimator.predict_proba(Xn.astype(np.float32, copy=False))[:, 1]
                    except Exception as exc:
                        logging.warning("[RawFallback] failed to fit GBDT gate: %s", exc)
                        estimator = None
                        scores = np.full(n_samples, -np.inf, dtype=np.float64)
                elif gate_type == "random_forest":
                    try:
                        from sklearn.ensemble import RandomForestClassifier

                        estimator = RandomForestClassifier(
                            n_estimators=int(getattr(self, "fallback_gate_tree_estimators", 300)),
                            max_depth=int(getattr(self, "fallback_gate_tree_max_depth", 4)),
                            min_samples_leaf=int(
                                getattr(self, "fallback_gate_tree_min_samples_leaf", 10)
                            ),
                            class_weight="balanced_subsample",
                            random_state=0,
                            n_jobs=4,
                        )
                        estimator.fit(Xn.astype(np.float32, copy=False), y.astype(np.int64))
                        scores = estimator.predict_proba(Xn.astype(np.float32, copy=False))[:, 1]
                    except Exception as exc:
                        logging.warning("[RawFallback] failed to fit random-forest gate: %s", exc)
                        estimator = None
                        scores = np.full(n_samples, -np.inf, dtype=np.float64)
                else:
                    scores = np.full(n_samples, -np.inf, dtype=np.float64)

                threshold, _ = self._select_raw_fallback_gate_threshold(
                    scores=scores,
                    compact_correct=compact_correct,
                    fallback_correct=fallback_correct,
                )
            gate_mask = np.isfinite(scores) & (scores >= threshold)
            threshold_stats = {
                "fallback_rate": float(np.mean(gate_mask)) if n_samples > 0 else 0.0,
                "net_gain_rate": float(
                    np.mean(np.where(gate_mask, fallback_correct, compact_correct)) - np.mean(compact_correct)
                ) if n_samples > 0 else 0.0,
                "net_gain_count": float(np.sum(gate_mask & benefit) - np.sum(gate_mask & harm)),
                "benefit_selected": float(np.sum(gate_mask & benefit)),
                "harm_selected": float(np.sum(gate_mask & harm)),
                "neutral_selected": float(np.sum(gate_mask & neutral)),
                "benefit_recall": float(np.sum(gate_mask & benefit)) / max(1.0, float(np.sum(benefit))),
                "utility_precision": float(np.sum(gate_mask & benefit)) / max(
                    1.0,
                    float(np.sum(gate_mask & benefit) + np.sum(gate_mask & harm)),
                ),
                "selected_budget_rate": float(getattr(self, "fallback_gate_max_rate", 1.0)),
            }
        else:
            train_mask = candidate_mask.astype(bool)
            scores = np.full(n_samples, -np.inf, dtype=np.float64)
            if gate_type == "candidate_rule":
                compact_margin_safe = np.maximum(compact_margin.astype(np.float64), 1e-6)
                raw_ratio = np.nan_to_num(fallback_margin.astype(np.float64) / compact_margin_safe, nan=0.0)
                scores = np.where(
                    candidate_mask,
                    score_cross_gap.astype(np.float64) + 0.05 * raw_ratio,
                    -np.inf,
                )
            elif gate_type in ("delta_ridge", "delta_logistic", "delta_ridge_pair"):
                y = delta_target.astype(np.float64, copy=False)
                sample_weight = np.full(n_samples, float(getattr(self, "fallback_gate_neutral_weight", 0.25)), dtype=np.float64)
                sample_weight[benefit | harm] = 1.0
                sample_weight[~train_mask] = 0.0
                if np.sum(sample_weight > 0.0) > 0:
                    Xaug = np.concatenate(
                        [
                            np.ones((n_samples, 1), dtype=np.float64),
                            Xn.astype(np.float64, copy=False),
                        ],
                        axis=1,
                    )
                    sqrt_w = np.sqrt(sample_weight)[:, None]
                    Xw = Xaug * sqrt_w
                    yw = y * np.sqrt(sample_weight)
                    reg_lambda = 1.0
                    gram = Xw.T @ Xw
                    reg = np.eye(gram.shape[0], dtype=np.float64) * reg_lambda
                    reg[0, 0] = 0.0
                    try:
                        coef = np.linalg.solve(gram + reg, Xw.T @ yw)
                    except np.linalg.LinAlgError:
                        coef = np.linalg.lstsq(gram + reg, Xw.T @ yw, rcond=None)[0]
                    bias = float(coef[0])
                    weights = coef[1:]
                    scores = Xaug @ coef
                    scores[~candidate_mask] = -np.inf
            elif gate_type == "benefit_harm_logistic":
                train_mask = candidate_mask & (benefit | harm)
                if np.sum(train_mask) > 0 and len(np.unique(benefit[train_mask].astype(np.int64))) == 2:
                    try:
                        from sklearn.linear_model import LogisticRegression

                        estimator = LogisticRegression(
                            class_weight="balanced",
                            max_iter=500,
                            solver="liblinear",
                            random_state=0,
                        )
                        estimator.fit(
                            Xn[train_mask].astype(np.float32, copy=False),
                            benefit[train_mask].astype(np.int64),
                        )
                        scores = 2.0 * estimator.predict_proba(Xn.astype(np.float32, copy=False))[:, 1] - 1.0
                        scores[~candidate_mask] = -np.inf
                    except Exception as exc:
                        logging.warning("[RawFallback] failed to fit benefit-harm logistic gate: %s", exc)
                        estimator = None
            elif gate_type == "dual_logistic":
                try:
                    from sklearn.linear_model import LogisticRegression

                    compact_feature_names = {
                        "compact_margin",
                        "compact_top1_score",
                        "compact_top2_score",
                        "selected_residual",
                        "selected_distance",
                        "selected_count_log",
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
                        "compact_raw_disagree",
                        "score_cross_gap",
                        "compact_score_on_raw_top1",
                        "raw_score_on_compact_top1",
                    }
                    raw_feature_names = {
                        "fallback_distance",
                        "fallback_margin",
                        "raw_top1_score",
                        "raw_top2_score",
                        "fallback_count_log",
                        "fallback_residual",
                        "raw_rank_of_compact_top1",
                        "raw_class_age",
                        "fallback_available",
                        "compact_raw_disagree",
                        "score_cross_gap",
                        "compact_score_on_raw_top1",
                        "raw_score_on_compact_top1",
                    }
                    compact_feature_indices = [
                        i for i, name in enumerate(feature_names) if name in compact_feature_names
                    ]
                    raw_feature_indices = [
                        i for i, name in enumerate(feature_names) if name in raw_feature_names
                    ]
                    if not compact_feature_indices:
                        compact_feature_indices = list(range(n_features))
                    if not raw_feature_indices:
                        raw_feature_indices = list(range(n_features))
                    X_compact = Xn[:, compact_feature_indices].astype(np.float32, copy=False)
                    X_raw = Xn[:, raw_feature_indices].astype(np.float32, copy=False)
                    compact_estimator = LogisticRegression(
                        class_weight="balanced",
                        max_iter=500,
                        solver="liblinear",
                        random_state=0,
                    )
                    raw_estimator = LogisticRegression(
                        class_weight="balanced",
                        max_iter=500,
                        solver="liblinear",
                        random_state=1,
                    )
                    if len(np.unique(compact_correct.astype(np.int64))) == 2:
                        compact_estimator.fit(
                            X_compact,
                            compact_correct.astype(np.int64),
                        )
                        p_compact = compact_estimator.predict_proba(
                            X_compact
                        )[:, 1]
                    else:
                        p_compact = np.full(n_samples, float(np.mean(compact_correct)), dtype=np.float64)
                        compact_estimator = None
                    if len(np.unique(fallback_correct.astype(np.int64))) == 2:
                        raw_estimator.fit(
                            X_raw,
                            fallback_correct.astype(np.int64),
                        )
                        p_raw = raw_estimator.predict_proba(
                            X_raw
                        )[:, 1]
                    else:
                        p_raw = np.full(n_samples, float(np.mean(fallback_correct)), dtype=np.float64)
                        raw_estimator = None
                    scores = p_raw - p_compact
                    scores[~candidate_mask] = -np.inf
                except Exception as exc:
                    logging.warning("[RawFallback] failed to fit dual-logistic gate: %s", exc)
                    compact_estimator = None
                    raw_estimator = None
                    compact_feature_indices = []
                    raw_feature_indices = []
            elif gate_type == "utility_tree":
                if np.sum(train_mask) > 0:
                    try:
                        from sklearn.tree import DecisionTreeRegressor

                        estimator = DecisionTreeRegressor(
                            max_depth=int(getattr(self, "fallback_gate_tree_max_depth", 2)),
                            min_samples_leaf=int(getattr(self, "fallback_gate_tree_min_samples_leaf", 12)),
                            random_state=0,
                        )
                        sample_weight = np.full(n_samples, float(getattr(self, "fallback_gate_neutral_weight", 0.25)), dtype=np.float64)
                        sample_weight[benefit | harm] = 1.0
                        estimator.fit(
                            Xn[train_mask].astype(np.float32, copy=False),
                            delta_target[train_mask].astype(np.float32),
                            sample_weight=sample_weight[train_mask],
                        )
                        scores = estimator.predict(Xn.astype(np.float32, copy=False))
                        scores[~candidate_mask] = -np.inf
                    except Exception as exc:
                        logging.warning("[RawFallback] failed to fit utility tree gate: %s", exc)
                        estimator = None
            elif gate_type == "utility_gbdt":
                if np.sum(train_mask) > 0:
                    try:
                        from sklearn.ensemble import GradientBoostingRegressor

                        estimator = GradientBoostingRegressor(
                            n_estimators=int(getattr(self, "fallback_gate_tree_estimators", 160)),
                            learning_rate=0.04,
                            max_depth=int(getattr(self, "fallback_gate_tree_max_depth", 2)),
                            min_samples_leaf=int(getattr(self, "fallback_gate_tree_min_samples_leaf", 12)),
                            random_state=0,
                        )
                        sample_weight = np.full(n_samples, float(getattr(self, "fallback_gate_neutral_weight", 0.25)), dtype=np.float64)
                        sample_weight[benefit | harm] = 1.0
                        estimator.fit(
                            Xn[train_mask].astype(np.float32, copy=False),
                            delta_target[train_mask].astype(np.float32),
                            sample_weight=sample_weight[train_mask],
                        )
                        scores = estimator.predict(Xn.astype(np.float32, copy=False))
                        scores[~candidate_mask] = -np.inf
                    except Exception as exc:
                        logging.warning("[RawFallback] failed to fit utility GBDT gate: %s", exc)
                        estimator = None
            elif gate_type == "utility_table":
                scores = np.zeros(n_samples, dtype=np.float64)
                scores[~candidate_mask] = -np.inf

            if pair_mode not in ("", "none"):
                pair_tables = self._fit_fallback_utility_tables(
                    compact_top1=compact_top1,
                    fallback_top1=fallback_top1,
                    selected_node=selected_node,
                    task_relation_code=task_relation_code,
                    compact_correct=compact_correct,
                    fallback_correct=fallback_correct,
                )
                pair_scores, node_scores = self._lookup_fallback_utility_tables(
                    pair_tables,
                    compact_top1=compact_top1,
                    fallback_top1=fallback_top1,
                    selected_node=selected_node,
                    task_relation_code=task_relation_code,
                )
                mode = pair_mode
                if mode in ("class_pair", "class_pair_only"):
                    scores = scores + float(getattr(self, "fallback_pair_table_lambda", 0.5)) * pair_scores
                elif mode in ("class_pair_task", "class_pair_task_only"):
                    scores = scores + float(getattr(self, "fallback_pair_table_lambda", 0.5)) * pair_scores
                elif mode in ("node_pair", "node_pair_only"):
                    scores = scores + float(getattr(self, "fallback_node_table_lambda", 0.5)) * node_scores
                elif mode in ("full", "pair_node", "node_pair_full"):
                    scores = scores + float(getattr(self, "fallback_pair_table_lambda", 0.5)) * pair_scores
                    scores = scores + float(getattr(self, "fallback_node_table_lambda", 0.5)) * node_scores
                scores[~candidate_mask] = -np.inf

            threshold, gate_mask, threshold_stats = self._select_raw_fallback_gate_threshold_budgeted(
                scores=scores,
                compact_correct=compact_correct,
                fallback_correct=fallback_correct,
                candidate_mask=candidate_mask,
            )

        if gate_type not in utility_gate_types:
            gate_mask = np.isfinite(scores) & (scores >= threshold)
        final_correct = np.where(gate_mask, fallback_correct, compact_correct)
        compact_accuracy = float(np.mean(compact_correct)) if n_samples > 0 else 0.0
        calibration_accuracy = float(np.mean(final_correct)) if n_samples > 0 else 0.0
        calibration_gain = calibration_accuracy - compact_accuracy
        gate_disabled = False
        gate_disabled_reason = ""
        if pos_count < float(getattr(self, "fallback_gate_min_positives", 0)):
            gate_disabled = True
            gate_disabled_reason = "insufficient_positives"
        elif calibration_gain < float(getattr(self, "fallback_gate_min_calibration_gain", 0.0)):
            gate_disabled = True
            gate_disabled_reason = "insufficient_calibration_gain"

        if gate_disabled:
            threshold = float("inf")
            gate_mask = np.zeros(n_samples, dtype=bool)
            final_correct = compact_correct
            calibration_accuracy = compact_accuracy
            calibration_gain = 0.0
            threshold_stats = dict(threshold_stats)
            threshold_stats.update({
                "fallback_rate": 0.0,
                "net_gain_rate": 0.0,
                "net_gain_count": 0.0,
                "benefit_selected": 0.0,
                "harm_selected": 0.0,
                "neutral_selected": 0.0,
                "benefit_recall": 0.0,
                "utility_precision": 0.0,
            })

        self._raw_fallback_gate_model = {
            "type": gate_type,
            "objective": "oracle_positive_binary" if gate_type not in utility_gate_types else "delta_utility",
            "feature_names": list(feature_names),
            "feature_mean": np.asarray(feature_mean, dtype=np.float32),
            "feature_std": np.asarray(feature_std, dtype=np.float32),
            "weights": np.asarray(weights, dtype=np.float32),
            "bias": float(bias),
            "threshold": float(threshold),
            "estimator": estimator,
            "compact_estimator": compact_estimator,
            "raw_estimator": raw_estimator,
            "compact_feature_indices": np.asarray(compact_feature_indices, dtype=np.int64),
            "raw_feature_indices": np.asarray(raw_feature_indices, dtype=np.int64),
            "candidate_rule": candidate_rule,
            "candidate_params": dict(candidate_params),
            "budget_rates": tuple(getattr(self, "fallback_gate_budget_rates", ())),
            "threshold_strategy": str(getattr(self, "fallback_gate_threshold_strategy", "max_net")),
            "pair_table_mode": pair_mode,
            "fallback_pair_table_lambda": float(getattr(self, "fallback_pair_table_lambda", 0.5)),
            "fallback_node_table_lambda": float(getattr(self, "fallback_node_table_lambda", 0.5)),
            "disabled": bool(gate_disabled),
            "disabled_reason": gate_disabled_reason,
        }
        self._raw_fallback_gate_model.update(pair_tables)
        self._raw_fallback_gate_model_dirty = False
        fit_stats = {
            "gate_type": gate_type,
            "objective": 0.0 if gate_type not in utility_gate_types else 1.0,
            "samples": float(n_samples),
            "positives": float(pos_count),
            "negatives": float(neg_count),
            "positive_rate": float(np.mean(gate_target > 0.5)) if n_samples > 0 else 0.0,
            "benefit_count": float(np.sum(benefit)),
            "harm_count": float(np.sum(harm)),
            "both_correct_count": float(np.sum(both_correct)),
            "both_wrong_count": float(np.sum(both_wrong)),
            "neutral_count": float(np.sum(neutral)),
            "benefit_rate": float(np.mean(benefit)) if n_samples > 0 else 0.0,
            "harm_rate": float(np.mean(harm)) if n_samples > 0 else 0.0,
            "both_correct_rate": float(np.mean(both_correct)) if n_samples > 0 else 0.0,
            "both_wrong_rate": float(np.mean(both_wrong)) if n_samples > 0 else 0.0,
            "neutral_rate": float(np.mean(neutral)) if n_samples > 0 else 0.0,
            "oracle_gain": float(np.mean(benefit)) if n_samples > 0 else 0.0,
            "calibration_samples": float(n_samples),
            "calibration_positive_rate": float(np.mean(gate_target > 0.5)) if n_samples > 0 else 0.0,
            "calibration_pos_weight": float(pos_weight),
            "compact_accuracy": compact_accuracy,
            "fallback_accuracy": float(np.mean(fallback_correct)) if n_samples > 0 else 0.0,
            "oracle_accuracy": float(np.mean(np.logical_or(compact_correct, fallback_correct))) if n_samples > 0 else 0.0,
            "calibration_compact_acc": compact_accuracy,
            "calibration_fallback_acc": float(np.mean(fallback_correct)) if n_samples > 0 else 0.0,
            "calibration_accuracy": calibration_accuracy,
            "calibration_final_acc": calibration_accuracy,
            "calibration_gain": calibration_gain,
            "calibration_fallback_rate": float(np.mean(gate_mask)) if n_samples > 0 else 0.0,
            "calibration_net_gain": float(threshold_stats.get("net_gain_rate", calibration_gain)),
            "calibration_net_gain_lcb": float(threshold_stats.get("net_gain_lcb", 0.0)),
            "calibration_net_gain_std": float(threshold_stats.get("net_gain_std", 0.0)),
            "calibration_net_gain_count": float(threshold_stats.get("net_gain_count", 0.0)),
            "calibration_benefit_selected": float(threshold_stats.get("benefit_selected", 0.0)),
            "calibration_harm_selected": float(threshold_stats.get("harm_selected", 0.0)),
            "calibration_neutral_selected": float(threshold_stats.get("neutral_selected", 0.0)),
            "calibration_benefit_recall": float(threshold_stats.get("benefit_recall", 0.0)),
            "calibration_utility_precision": float(threshold_stats.get("utility_precision", 0.0)),
            "calibration_selected_budget_rate": float(threshold_stats.get("selected_budget_rate", 0.0)),
            "candidate_rule": candidate_rule,
            "candidate_rate": float(np.mean(candidate_mask)) if n_samples > 0 else 0.0,
            "candidate_benefit_rate": float(np.mean(benefit[candidate_mask])) if np.any(candidate_mask) else 0.0,
            "candidate_harm_rate": float(np.mean(harm[candidate_mask])) if np.any(candidate_mask) else 0.0,
            "candidate_net_gain": float(
                (np.sum(benefit[candidate_mask]) - np.sum(harm[candidate_mask])) / max(1, n_samples)
            ),
            "threshold": float(threshold),
            "weight_l2": float(np.linalg.norm(weights)),
            "calibration_threshold": float(threshold),
            "calibration_weight_norm": float(np.linalg.norm(weights)),
            "calibration_bias": float(bias),
            "fallback_gate_neutral_weight": float(getattr(self, "fallback_gate_neutral_weight", 0.25)),
            "fallback_pair_table_mode": 0.0 if pair_mode in ("", "none") else 1.0,
            "fallback_pair_table_entries": float(len(pair_tables.get("class_pair_table", {}))) if pair_tables else 0.0,
            "fallback_pair_task_table_entries": float(len(pair_tables.get("class_pair_task_table", {}))) if pair_tables else 0.0,
            "fallback_node_table_entries": float(len(pair_tables.get("node_pair_table", {}))) if pair_tables else 0.0,
            "fallback_gate_max_rate": float(getattr(self, "fallback_gate_max_rate", 1.0)),
            "fallback_gate_tree_estimators": float(getattr(self, "fallback_gate_tree_estimators", 0)),
            "fallback_gate_tree_max_depth": float(getattr(self, "fallback_gate_tree_max_depth", 0)),
            "fallback_gate_tree_min_samples_leaf": float(
                getattr(self, "fallback_gate_tree_min_samples_leaf", 0)
            ),
            "fallback_gate_threshold_strategy": 1.0
            if str(getattr(self, "fallback_gate_threshold_strategy", "max_net")).lower() in (
                "lcb",
                "lower_confidence_bound",
                "conservative_lcb",
            )
            else 0.0,
            "fallback_gate_min_precision": float(getattr(self, "fallback_gate_min_precision", 0.0)),
            "fallback_gate_min_selected": float(getattr(self, "fallback_gate_min_selected", 0)),
            "fallback_gate_min_net_count": float(getattr(self, "fallback_gate_min_net_count", 0)),
            "gate_disabled": float(gate_disabled),
            "fallback_gate_min_positives": float(getattr(self, "fallback_gate_min_positives", 0)),
            "fallback_gate_min_calibration_gain": float(
                getattr(self, "fallback_gate_min_calibration_gain", 0.0)
            ),
        }
        if gate_disabled_reason:
            fit_stats["gate_disabled_reason"] = gate_disabled_reason
        self._raw_fallback_gate_fit_stats = fit_stats
        logging.info(
            "[RawFallback] calibrated learned gate: type=%s, samples=%d, pos_rate=%.4f, "
            "benefit=%.4f, harm=%.4f, candidate_rate=%.4f, compact_acc=%.4f, "
            "fallback_acc=%.4f, final_acc=%.4f, gain=%.4f, net=%.4f, "
            "fallback_rate=%.4f, precision=%.4f, threshold=%.6f, disabled=%d",
            gate_type,
            int(n_samples),
            fit_stats["calibration_positive_rate"],
            fit_stats["benefit_rate"],
            fit_stats["harm_rate"],
            fit_stats["candidate_rate"],
            fit_stats["calibration_compact_acc"],
            fit_stats["calibration_fallback_acc"],
            fit_stats["calibration_final_acc"],
            fit_stats["calibration_gain"],
            fit_stats["calibration_net_gain"],
            fit_stats["calibration_fallback_rate"],
            fit_stats["calibration_utility_precision"],
            fit_stats["calibration_threshold"],
            int(fit_stats["gate_disabled"]),
        )
        return fit_stats

    def _summarize_raw_fallback_eval_stats(self) -> Dict[str, float]:
        stats = getattr(self, "_raw_fallback_eval_stats", {})
        samples = float(stats.get("samples", 0.0))
        used = float(stats.get("fallback_used", 0.0))
        available = float(stats.get("fallback_available", 0.0))
        nodes, classes = self._fallback_cache_counts()
        out = {
            "fallback_enabled": float(bool(getattr(self, "use_raw_fallback_gate", False))),
            "fallback_cache_nodes": float(nodes),
            "fallback_cache_classes": float(classes),
            "fallback_samples": samples,
            "fallback_rate": used / samples if samples > 0 else 0.0,
            "fallback_available_rate": available / samples if samples > 0 else 0.0,
            "fallback_prediction_change_rate": (
                float(stats.get("prediction_changed", 0.0)) / samples if samples > 0 else 0.0
            ),
            "compact_raw_disagreement": (
                float(stats.get("compact_raw_disagree", 0.0)) / samples if samples > 0 else 0.0
            ),
            "compact_accuracy": float(stats.get("compact_correct", 0.0)) / samples if samples > 0 else 0.0,
            "final_accuracy": float(stats.get("final_correct", 0.0)) / samples if samples > 0 else 0.0,
            "fallback_accuracy": (
                float(stats.get("fallback_correct", 0.0)) / available if available > 0 else 0.0
            ),
            "compact_correct_when_used": (
                float(stats.get("compact_correct_when_used", 0.0)) / used if used > 0 else 0.0
            ),
            "fallback_correct_when_used": (
                float(stats.get("fallback_correct_when_used", 0.0)) / used if used > 0 else 0.0
            ),
            "fallback_margin_mean": (
                float(stats.get("fallback_margin_sum", 0.0)) / used if used > 0 else 0.0
            ),
            "fallback_residual_mean": (
                float(stats.get("fallback_residual_sum", 0.0)) / used if used > 0 else 0.0
            ),
            "trace_margin_mean": float(stats.get("margin_sum", 0.0)) / samples if samples > 0 else 0.0,
            "trace_residual_mean": float(stats.get("residual_sum", 0.0)) / samples if samples > 0 else 0.0,
            "oracle_improvable_rate": (
                float(stats.get("oracle_improvable", 0.0)) / samples if samples > 0 else 0.0
            ),
            "benefit_rate": float(stats.get("benefit_total", 0.0)) / samples if samples > 0 else 0.0,
            "harm_rate": float(stats.get("harm_total", 0.0)) / samples if samples > 0 else 0.0,
            "neutral_rate": float(stats.get("neutral_total", 0.0)) / samples if samples > 0 else 0.0,
            "benefit_selected": float(stats.get("benefit_selected", 0.0)),
            "harm_selected": float(stats.get("harm_selected", 0.0)),
            "neutral_selected": float(stats.get("neutral_selected", 0.0)),
            "net_gain_count": float(stats.get("benefit_selected", 0.0)) - float(stats.get("harm_selected", 0.0)),
            "net_gain_rate": (
                (float(stats.get("benefit_selected", 0.0)) - float(stats.get("harm_selected", 0.0))) / samples
                if samples > 0 else 0.0
            ),
            "benefit_recall": (
                float(stats.get("benefit_selected", 0.0)) / float(stats.get("benefit_total", 0.0))
                if float(stats.get("benefit_total", 0.0)) > 0 else 0.0
            ),
            "utility_precision": (
                float(stats.get("benefit_selected", 0.0))
                / max(1.0, float(stats.get("benefit_selected", 0.0)) + float(stats.get("harm_selected", 0.0)))
            ),
        }
        return out

    # ================================================================== #
    # LifeTopoDict: Dictionary initialisation, coding, materialisation   #
    # ================================================================== #

    def dict_init(self) -> None:
        """Initialise the global shared dictionary from class-means.

        Collects all ``class_mu_raw`` vectors (raw / unnormalised class means),
        L2-normalises each one to form the initial atom bank, and sets all
        atom states to ``'plastic'``.

        This method should be called **after** the base-task ``compress()`` so
        that every class already has a valid ``class_mu_raw`` entry.
        """
        if not self.use_dict_coding:
            logging.info("[LifeTopoDict] dict_init() skipped (use_dict_coding=False).")
            return

        # --- Collect raw class means ---
        class_ids = sorted(self.class_mu_raw.keys())
        if len(class_ids) == 0:
            logging.warning("[LifeTopoDict] dict_init(): no class_mu_raw available; skipping.")
            return

        atom_vectors: List[np.ndarray] = []
        for cls in class_ids:
            mu_raw = self.class_mu_raw[cls].astype(np.float32, copy=True)
            atom_vectors.append(mu_raw)

        atoms = np.stack(atom_vectors, axis=0).astype(np.float32)  # [M_0, d]
        M_0, d = atoms.shape

        # --- L2 normalise each atom row ---
        norms = np.linalg.norm(atoms, axis=1, keepdims=True).clip(min=1e-8)
        atoms_hat = (atoms / norms).astype(np.float32)

        # Both dict_atoms and dict_atoms_hat store normalised atoms.
        # dict_atoms is kept for backward compat (center_raw = D @ a).
        self.dict_atoms = atoms_hat.copy()
        self.dict_atoms_hat = atoms_hat
        self.atom_states = ['plastic'] * M_0
        self.atom_origins = ['base'] * M_0
        self.atom_usage = {m: 0.0 for m in range(M_0)}
        self.atom_usage_ema = {m: 0.0 for m in range(M_0)}
        self._dict_initialized = True

        logging.info(
            f"[LifeTopoDict] dict_init(): initialised dictionary with "
            f"M={M_0} atoms, d={d}."
        )

    def sparse_encode(
        self,
        v: np.ndarray,
        k: Optional[int] = None,
        lambda_ridge: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Top-K Ridge sparse encoding against the current dictionary.

        Args:
            v: Target vector(s).  Shape ``(d,)`` for single or ``(N, d)`` for batch.
            k: Number of atoms in the support set.  Defaults to ``self.dict_sparse_k``.
            lambda_ridge: Ridge regularisation strength.  Defaults to ``self.dict_ridge_lambda``.

        Returns:
            coeff: Sparse coefficient vector(s).  Shape ``(M,)`` or ``(N, M)``.
            residual: Cosine reconstruction residual ``1 - cos(v, D @ a)``.
                      Scalar for single input, or ``(N,)`` for batch.
        """
        if self.dict_atoms_hat is None:
            raise RuntimeError("sparse_encode() called before dict_init().")

        k = k if k is not None else self.dict_sparse_k
        lambda_ridge = lambda_ridge if lambda_ridge is not None else self.dict_ridge_lambda

        D_hat = self.dict_atoms_hat  # [M, d]
        M, d = D_hat.shape
        k = min(k, M)

        single = (v.ndim == 1)
        if single:
            v = v[np.newaxis, :]  # [1, d]

        N = v.shape[0]

        # Build active mask: exclude inactive atoms
        active_mask = np.array([s != 'inactive' for s in self.atom_states], dtype=bool)
        if not np.any(active_mask):
            # Degenerate case: all inactive
            coeff_out = np.zeros((N, M), dtype=np.float32)
            residual_out = np.ones(N, dtype=np.float32)
            if single:
                return coeff_out[0], float(residual_out[0])
            return coeff_out, residual_out

        active_indices = np.where(active_mask)[0]
        D_active = D_hat[active_indices]  # [M_act, d]
        M_act = D_active.shape[0]
        k_act = min(k, M_act)

        # L2-normalise target vectors
        v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)  # [N, d]

        # Cosine similarity: v_norm @ D_active^T  => [N, M_act]
        cos_sim = v_norm @ D_active.T

        # Top-k indices per row (handle case where k_act >= M_act)
        if k_act >= M_act:
            # Use all active atoms directly
            top_k_indices = np.tile(np.arange(M_act), (N, 1))  # [N, M_act]
        else:
            top_k_indices = np.argpartition(-cos_sim, kth=k_act, axis=1)[:, :k_act]  # [N, k_act]

        # Solve ridge regression per sample on the support set
        coeff_full = np.zeros((N, M_act), dtype=np.float32)

        for i in range(N):
            support = top_k_indices[i]  # [k_act]
            D_S = D_active[support]     # [k_act, d]

            # a = (D_S^T D_S + lambda * I)^{-1} D_S^T v_i
            gram = D_S @ D_S.T + lambda_ridge * np.eye(len(support), dtype=np.float32)
            try:
                a_support = np.linalg.solve(gram, D_S @ v[i])
            except np.linalg.LinAlgError:
                a_support = np.linalg.lstsq(gram, D_S @ v[i], rcond=None)[0]
            coeff_full[i, support] = a_support.astype(np.float32)

        # Map active coefficients back to full M-length vector
        coeff_out = np.zeros((N, M), dtype=np.float32)
        coeff_out[:, active_indices] = coeff_full

        # Reconstruct and compute cosine residual
        recon = coeff_out @ D_hat  # [N, d]
        recon_norm = recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-8)
        cos_res = np.sum(v_norm * recon_norm, axis=1)  # [N]
        residual_out = (1.0 - cos_res).astype(np.float32)

        if single:
            return coeff_out[0], float(residual_out[0])
        return coeff_out, residual_out

    def _ensure_coeff_width(self, node: _Cluster, M: int) -> None:
        """Pad or trim an existing node coefficient vector to the atom count."""
        if node.coeff is None:
            return

        coeff = np.asarray(node.coeff, dtype=np.float32)
        if coeff.ndim == 1:
            coeff = coeff.reshape(1, -1)

        width = coeff.shape[1]
        if width == M:
            node.coeff = coeff
            return

        resized = np.zeros((1, M), dtype=np.float32)
        copy_width = min(width, M)
        if copy_width > 0:
            resized[:, :copy_width] = coeff[:, :copy_width]
        node.coeff = resized

    def _is_frozen_dict_node(self, cls: int, node: _Cluster) -> bool:
        """Return True if a node's dictionary code/center must not be rewritten."""
        if not getattr(self, 'use_dict_coding', False):
            return False
        if node.coeff is None:
            return False

        known_classes = getattr(self, 'known_classes_before_task', None)
        is_old_class = known_classes is not None and int(cls) < int(known_classes)
        is_protected_node = getattr(node, 'node_state', 'plastic') == 'protected'
        return bool(is_old_class or is_protected_node)

    def _encode_nodes_with_dict(self) -> None:
        """Encode every active node centre with the shared dictionary.

        Iterates over ``self.class_clusters`` and calls ``sparse_encode()`` on
        each node's ``center_raw``.  Nodes whose ``node_state`` is
        ``'inactive'`` are skipped.

        Per-task atom usage is collected, then EMA-updated into
        ``self.atom_usage_ema`` for lifecycle decisions.  The cumulative
        ``self.atom_usage`` is retained for diagnostics.
        """
        if not self._dict_initialized or self.dict_atoms_hat is None:
            logging.warning("[LifeTopoDict] _encode_nodes_with_dict(): dictionary not initialised.")
            return

        M = self.dict_atoms_hat.shape[0]

        # Reset per-task counters
        self.atom_usage_task = {m: 0.0 for m in range(M)}

        total_encoded = 0
        frozen_skipped = 0
        for cls, clusters in self.class_clusters.items():
            for node in clusters:
                if node.node_state == 'inactive':
                    continue
                self._ensure_coeff_width(node, M)
                if self._is_frozen_dict_node(cls, node):
                    frozen_skipped += 1
                    continue
                coeff, residual = self.sparse_encode(node.center_raw)
                node.coeff = coeff.reshape(1, -1)  # (1, M)
                node.residual = residual
                total_encoded += 1

                # Per-task usage counting
                support_indices = np.nonzero(np.abs(coeff) > 1e-6)[0]
                for idx in support_indices:
                    i = int(idx)
                    self.atom_usage_task[i] = self.atom_usage_task.get(i, 0.0) + 1.0
                    # Cumulative counter retained for diagnostics
                    self.atom_usage[i] = self.atom_usage.get(i, 0.0) + 1.0

        # EMA update: convert per-task counts to rates, then blend
        if total_encoded > 0:
            decay = self._usage_ema_decay
            for m in range(M):
                task_rate = self.atom_usage_task.get(m, 0.0) / float(total_encoded)
                old = self.atom_usage_ema.get(m, 0.0)
                self.atom_usage_ema[m] = decay * old + (1.0 - decay) * task_rate

        logging.info(
            f"[LifeTopoDict] _encode_nodes_with_dict(): encoded {total_encoded} nodes, "
            f"frozen_skipped={frozen_skipped}."
        )

    def _materialize_nodes(self) -> None:
        """Reconstruct node centres from the dictionary and sparse codes.

        Reconstruction semantics (unified):

        ``z = D_hat @ a``  where D_hat rows are L2-normalised.
        ``v_hat = normalize(z)``
        ``cluster.center = v_hat``  (L2-normalised, as HC-SOINN expects)
        ``cluster.dict_recon_raw = z``  (unnormalised reconstruction)

        ``cluster.center_raw`` is deliberately left untouched: it remains the
        true raw cluster mean and the future sparse-coding / growth target.
        """
        if not self._dict_initialized or self.dict_atoms_hat is None:
            return

        D_hat = self.dict_atoms_hat  # [M, d] — single source of truth

        total_materialized = 0
        frozen_skipped = 0
        raw_fallback_candidates: List[Dict[str, object]] = []
        capture_raw_fallback = bool(
            (getattr(self, 'use_raw_fallback_gate', False) or getattr(self, 'enable_prediction_trace', False))
            and getattr(self, 'raw_fallback_per_class', 0) > 0
        )
        source_task = -1
        if getattr(self, 'known_classes_before_task', None) is not None:
            source_task = int(self.known_classes_before_task)
        for cls, clusters in self.class_clusters.items():
            for node_idx, node in enumerate(clusters):
                if node.node_state == 'inactive':
                    continue
                if node.coeff is None:
                    continue
                self._ensure_coeff_width(node, D_hat.shape[0])
                if self._is_frozen_dict_node(cls, node):
                    frozen_skipped += 1
                    continue

                if capture_raw_fallback and isinstance(node.center_raw, np.ndarray):
                    raw_fallback_candidates.append({
                        "center_raw": node.center_raw.astype(np.float32, copy=True),
                        "class_id": int(cls),
                        "node_id": int(node_idx),
                        "count": int(node.count),
                        "residual": float(node.residual),
                        "node_state": str(getattr(node, "node_state", "plastic")),
                        "source_task": int(source_task),
                        "snapshot_version": int(self._raw_fallback_cache_version + 1),
                    })

                a = node.coeff.ravel()  # (M,)
                z = a @ D_hat           # [d]  unnormalised reconstruction
                z_norm = np.linalg.norm(z)
                if z_norm > 1e-8:
                    node.center = (z / z_norm).astype(np.float32)
                # else: degenerate — keep old centre
                if self.drop_node_raw_after_dict_materialize:
                    # In P0 deployment/inference, the materialized center plus
                    # sparse code are sufficient.  Keeping both the original
                    # raw target and D@a reconstruction triples node storage.
                    node.center_raw = None
                    node.dict_recon_raw = None
                else:
                    node.dict_recon_raw = z.astype(np.float32)
                total_materialized += 1

        if capture_raw_fallback:
            self._update_raw_fallback_cache(raw_fallback_candidates)

        logging.info(
            f"[LifeTopoDict] _materialize_nodes(): materialized {total_materialized} nodes, "
            f"frozen_skipped={frozen_skipped}."
        )
        self.invalidate_cache()

    # ================================================================== #
    # LifeTopoDict: Dictionary growth & new-class registration (Agent F) #
    # ================================================================== #

    def dict_grow_for_new_classes(self) -> None:
        """F-2: Register new classes that lack dictionary representation.

        When the dictionary is already initialised but new classes appear
        (i.e. classes in ``class_mu_raw`` that have no corresponding atom),
        this method adds the new class means as fresh atoms, pads all
        existing node coefficient vectors with zeros, and then performs
        sparse encoding on the new class nodes.
        """
        if not self.use_dict_coding or not self._dict_initialized:
            return
        if self.dict_atoms is None or self.dict_atoms_hat is None:
            return

        # Determine which classes already have an atom-like representation
        # by checking if any node in that class has been encoded.
        encoded_classes: Set[int] = set()
        for cls, clusters in self.class_clusters.items():
            for node in clusters:
                if node.coeff is not None:
                    encoded_classes.add(cls)
                    break

        # Classes that have class_mu_raw but no encoded nodes
        all_classes = set(self.class_mu_raw.keys())
        new_classes = all_classes - encoded_classes

        if not new_classes:
            return

        # Collect new atom vectors from class means
        new_atom_vectors: List[np.ndarray] = []
        for cls in sorted(new_classes):
            mu_raw = self.class_mu_raw[cls].astype(np.float32, copy=True)
            new_atom_vectors.append(mu_raw)

        if not new_atom_vectors:
            return

        new_atoms = np.stack(new_atom_vectors, axis=0).astype(np.float32)  # [G, d]
        G, d = new_atoms.shape

        # L2-normalise new atoms
        norms = np.linalg.norm(new_atoms, axis=1, keepdims=True).clip(min=1e-8)
        new_atoms_hat = (new_atoms / norms).astype(np.float32)

        # Append to existing dictionary (both stores normalised atoms)
        M_old = self.dict_atoms.shape[0]
        self.dict_atoms = np.concatenate([self.dict_atoms, new_atoms_hat], axis=0)
        self.dict_atoms_hat = np.concatenate([self.dict_atoms_hat, new_atoms_hat], axis=0)
        M_new = self.dict_atoms.shape[0]

        # Update atom metadata
        for m in range(M_old, M_new):
            self.atom_states.append('plastic')
            self.atom_origins.append('base')
            self.atom_usage[m] = 0.0
            self.atom_usage_ema[m] = 0.0
            self._low_usage_streak[m] = 0

        # Pad all existing node coefficient vectors with zeros
        for cls, clusters in self.class_clusters.items():
            for node in clusters:
                if node.coeff is not None:
                    old_coeff = node.coeff  # (1, M_old)
                    padded = np.zeros((1, M_new), dtype=np.float32)
                    padded[:, :M_old] = old_coeff
                    node.coeff = padded

        # Sparse-encode only the new class nodes
        for cls in sorted(new_classes):
            clusters = self.class_clusters.get(cls, [])
            for node in clusters:
                if node.node_state == 'inactive':
                    continue
                coeff, residual = self.sparse_encode(node.center_raw)
                node.coeff = coeff.reshape(1, -1)
                node.residual = residual
                # Update atom usage
                support_indices = np.nonzero(np.abs(coeff) > 1e-6)[0]
                for idx in support_indices:
                    self.atom_usage[int(idx)] = self.atom_usage.get(int(idx), 0.0) + 1.0

        logging.info(
            f"[LifeTopoDict] dict_grow_for_new_classes(): registered {G} new classes, "
            f"dictionary expanded from M={M_old} to M={M_new}."
        )

    def dict_grow(self) -> List[Dict[str, float]]:
        """F-1: Residual-triggered dictionary growth.

        After sparse encoding, checks for nodes whose reconstruction residual
        exceeds ``theta_residual``.  For each high-residual node the
        normalised residual direction is added as a new atom.  Growth is
        capped at ``max_growth_per_task`` per call.

        Returns:
            A list of per-growth-event dicts with keys
            ``'cls'``, ``'node_idx'``, ``'residual_before'``,
            ``'residual_after'`` (filled after re-encoding).
        """
        if not self.use_dict_coding or not self._dict_initialized:
            return []
        if self.dict_atoms_hat is None:
            return []

        # P0-4 ablation gate: skip residual-triggered growth entirely
        if not self.use_dictionary_growth:
            return []

        growth_events: List[Dict[str, float]] = []
        D_hat = self.dict_atoms_hat  # [M, d]
        M_old = D_hat.shape[0]

        # 1. Collect high-residual nodes
        high_residual_nodes: List[Tuple[int, int, np.ndarray, float]] = []  # (cls, idx, center_raw, residual)

        for cls, clusters in self.class_clusters.items():
            for idx, node in enumerate(clusters):
                if node.node_state == 'inactive':
                    continue
                self._ensure_coeff_width(node, M_old)
                if self._is_frozen_dict_node(cls, node):
                    continue
                if node.coeff is None:
                    continue
                if node.residual > self.theta_residual:
                    high_residual_nodes.append((cls, idx, node.center_raw, node.residual))

        if not high_residual_nodes:
            return []

        # Sort by residual descending so worst-case nodes get priority
        high_residual_nodes.sort(key=lambda x: x[3], reverse=True)

        # Cap growth
        n_grow = min(len(high_residual_nodes), self.max_growth_per_task)
        high_residual_nodes = high_residual_nodes[:n_grow]

        # 2. Compute residual directions for new atoms
        new_atom_vectors: List[np.ndarray] = []
        grow_node_info: List[Tuple[int, int, float]] = []  # (cls, idx, residual_before)

        for cls, idx, center_raw, residual in high_residual_nodes:
            node = self.class_clusters[cls][idx]
            self._ensure_coeff_width(node, M_old)
            a = node.coeff.ravel()  # (M,)

            # v_hat = D_hat^T @ a  (reconstruction)
            v_hat = a @ D_hat  # [d]

            # Residual direction: normalised (v - v_hat)
            v_norm = center_raw / (np.linalg.norm(center_raw) + 1e-8)
            v_hat_norm = v_hat / (np.linalg.norm(v_hat) + 1e-8)
            r_vec = v_norm - v_hat_norm
            r_norm = np.linalg.norm(r_vec)

            if r_norm > 1e-8:
                new_atom = (r_vec / r_norm).astype(np.float32)
                new_atom_vectors.append(new_atom)
                grow_node_info.append((cls, idx, residual))

        if not new_atom_vectors:
            return []

        # 3. Add new atoms to dictionary
        new_atoms_hat = np.stack(new_atom_vectors, axis=0).astype(np.float32)  # [G, d], already L2-norm
        G = new_atoms_hat.shape[0]

        # Both dict_atoms and dict_atoms_hat store normalised atoms
        self.dict_atoms = np.concatenate([self.dict_atoms, new_atoms_hat], axis=0)
        self.dict_atoms_hat = np.concatenate([self.dict_atoms_hat, new_atoms_hat], axis=0)
        M_new = self.dict_atoms.shape[0]

        # 4. Update atom metadata
        for m in range(M_old, M_new):
            self.atom_states.append('plastic')
            self.atom_origins.append('grown')
            self.atom_usage[m] = 0.0
            self.atom_usage_ema[m] = 0.0
            self._low_usage_streak[m] = 0

        # 5. Pad all existing coefficient vectors with zeros for the new dimensions
        for cls, clusters in self.class_clusters.items():
            for node in clusters:
                if node.coeff is not None:
                    old_coeff = node.coeff  # (1, M_old)
                    padded = np.zeros((1, M_new), dtype=np.float32)
                    padded[:, :M_old] = old_coeff
                    node.coeff = padded

        # 6. Re-encode only the grown nodes with the expanded dictionary
        for i, (cls, idx, residual_before) in enumerate(grow_node_info):
            node = self.class_clusters[cls][idx]
            if node.node_state == 'inactive':
                continue
            if self._is_frozen_dict_node(cls, node):
                continue

            # Record residual before for GTE
            coeff_new, residual_after = self.sparse_encode(node.center_raw)
            node.coeff = coeff_new.reshape(1, -1)
            node.residual = residual_after

            # Update atom usage for new encoding
            support_indices = np.nonzero(np.abs(coeff_new) > 1e-6)[0]
            for aidx in support_indices:
                self.atom_usage[int(aidx)] = self.atom_usage.get(int(aidx), 0.0) + 1.0

            growth_events.append({
                'cls': float(cls),
                'node_idx': float(idx),
                'residual_before': float(residual_before),
                'residual_after': float(residual_after),
            })

        logging.info(
            f"[LifeTopoDict] dict_grow(): added {G} atoms "
            f"(M {M_old} -> {M_new}) for {len(grow_node_info)} high-residual nodes."
        )

        return growth_events

    # ================================================================== #
    # LifeTopoDict: Diagnostic metrics (Agent F: F-3 .. F-6)             #
    # ================================================================== #

    def compute_PAD(self) -> float:
        """F-3: Compute Prototype Alignment Degree (PAD).

        For each class *c*, PAD_c = cos_sim(p_c, class_mu[c]) where p_c
        is the mean of the materialized prototypes for class *c*.

        Returns:
            Mean PAD across all classes (higher is better; max = 1.0).
        """
        if not self.use_dict_coding:
            return 0.0

        pad_values: List[float] = []

        for cls in self.class_clusters:
            clusters = self.class_clusters[cls]
            if not clusters:
                continue
            if cls not in self.class_mu:
                continue

            mu_c = self.class_mu[cls]  # normalised class mean

            # Collect materialized prototype centres for this class
            proto_centers: List[np.ndarray] = []
            for node in clusters:
                if node.node_state != 'inactive':
                    proto_centers.append(node.center)  # already normalised

            if not proto_centers:
                continue

            # Mean of prototypes (then normalise for cosine sim)
            proto_mean = np.mean(np.stack(proto_centers), axis=0)
            proto_mean_norm = proto_mean / (np.linalg.norm(proto_mean) + 1e-8)

            pad_c = float(np.dot(proto_mean_norm, mu_c))
            pad_values.append(pad_c)

        if not pad_values:
            return 0.0

        return float(np.mean(pad_values))

    def compute_eff_rank(self) -> float:
        """F-4: Compute atom-matrix effective rank.

        The guidance definition uses the entropy effective rank of the
        dictionary atom matrix singular values:

            exp(-sum_i p_i log p_i), where p_i = sigma_i / sum_j sigma_j.
        """
        if not self.use_dict_coding:
            return 0.0

        D_hat = self.dict_atoms_hat
        if D_hat is None or D_hat.size == 0:
            return 0.0

        try:
            singular_values = np.linalg.svd(D_hat.astype(np.float32, copy=False), compute_uv=False)
        except np.linalg.LinAlgError:
            singular_values = np.linalg.svd(D_hat.astype(np.float64, copy=False), compute_uv=False)

        singular_values = singular_values[singular_values > 1e-12]
        total = float(np.sum(singular_values))
        if total <= 1e-12 or singular_values.size == 0:
            return 0.0

        p = singular_values / total
        entropy = -float(np.sum(p * np.log(p + 1e-12)))
        return float(np.exp(entropy))

    def compute_coeff_eff_rank(self) -> float:
        """Compute the old coefficient effective-support diagnostic.

        This preserves the previous ``compute_eff_rank`` semantics under an
        explicit name so logs distinguish atom rank from sparse-code support.
        """
        if not self.use_dict_coding:
            return 0.0

        eff_ranks: List[float] = []

        for cls, clusters in self.class_clusters.items():
            for node in clusters:
                if node.node_state == 'inactive' or node.coeff is None:
                    continue
                a = node.coeff.ravel()
                abs_sum = np.sum(np.abs(a))
                sq_sum = np.sum(a ** 2)
                if sq_sum < 1e-12:
                    continue
                eff_rank_i = (abs_sum ** 2) / sq_sum
                eff_ranks.append(float(eff_rank_i))

        if not eff_ranks:
            return 0.0

        return float(np.mean(eff_ranks))

    def compute_GTE(self) -> float:
        """F-5: Compute Growth Trigger Effectiveness.

        GTE = mean of (residual_before - residual_after) / residual_before
        across all recorded growth events.

        Returns:
            Mean relative residual improvement ratio. Returns ``0.0`` when no
            growth events have been recorded yet, so diagnostics stay finite.
        """
        if not self._gte_records:
            return 0.0

        ratios: List[float] = []
        for rec in self._gte_records:
            rb = rec.get('residual_before', 0.0)
            ra = rec.get('residual_after', 0.0)
            if rb > 1e-8:
                ratios.append((rb - ra) / rb)
            else:
                ratios.append(0.0)

        if not ratios:
            return 0.0

        value = float(np.mean(ratios))
        return value if np.isfinite(value) else 0.0

    def _compute_edge_reliability(
        self,
        clusters: List[_Cluster],
        edges: Dict[int, Dict[int, int]],
    ) -> Dict[int, Dict[int, float]]:
        """Compute a simple auditable reliability score for persisted edges.

        Reliability is high for nearby endpoints and young edges. It is stored
        for diagnostics/P1 scoring, but it does not affect P0 inference.
        """
        reliability: Dict[int, Dict[int, float]] = {}
        if not clusters:
            return reliability

        for i, neighbors in edges.items():
            if i >= len(clusters):
                continue
            reliability[i] = {}
            ci = clusters[i].center
            for j, age in neighbors.items():
                if j >= len(clusters):
                    continue
                cj = clusters[j].center
                endpoint_sim = max(0.0, float(np.dot(ci, cj)))
                age_factor = 1.0 / (1.0 + float(max(age, 0)))
                reliability[i][j] = endpoint_sim * age_factor

        return reliability


    def _compute_memory_breakdown(self) -> Dict[str, float]:
        """Return per-component memory in bytes (uses .nbytes for numpy arrays).

        Returns:
            Dict with *_bytes and *_mb keys for each memory component,
            plus compact_deployable_bytes/mb and actual_implementation_bytes/mb.
        """
        breakdown: Dict[str, float] = {}

        def _array_nbytes(obj) -> float:
            if isinstance(obj, np.ndarray):
                return float(obj.nbytes)
            if isinstance(obj, torch.Tensor):
                return float(obj.numel() * obj.element_size())
            if isinstance(obj, dict):
                return float(sum(_array_nbytes(v) for v in obj.values()))
            if isinstance(obj, (list, tuple, set)):
                return float(sum(_array_nbytes(v) for v in obj))
            return 0.0

        # --- Atoms ---
        atom_deployable_bytes = 0.0
        if self.dict_atoms_hat is not None:
            atom_deployable_bytes = float(self.dict_atoms_hat.nbytes)
        elif self.dict_atoms is not None:
            atom_deployable_bytes = float(self.dict_atoms.nbytes)

        atom_actual_bytes = 0.0
        if self.dict_atoms is not None:
            atom_actual_bytes += float(self.dict_atoms.nbytes)
        if self.dict_atoms_hat is not None and self.dict_atoms_hat is not self.dict_atoms:
            atom_actual_bytes += float(self.dict_atoms_hat.nbytes)

        breakdown['atoms_bytes'] = atom_deployable_bytes
        breakdown['atoms_actual_bytes'] = atom_actual_bytes
        breakdown['atoms_duplicate_bytes'] = max(0.0, atom_actual_bytes - atom_deployable_bytes)

        # --- Coefficients (all nodes) ---
        coeff_bytes = 0.0
        for cls, clusters in self.class_clusters.items():
            for c in clusters:
                if c.coeff is not None and isinstance(c.coeff, np.ndarray):
                    coeff_bytes += c.coeff.nbytes
        breakdown['coefficients_bytes'] = coeff_bytes

        # --- Atom metadata (states, origins, usage strings/dicts) ---
        metadata_bytes = 0.0
        # atom_states: list of strings
        for s in self.atom_states:
            metadata_bytes += len(s.encode('utf-8'))
        # atom_origins: list of strings
        for s in getattr(self, 'atom_origins', []):
            metadata_bytes += len(s.encode('utf-8'))
        # atom_usage: dict[int, float] -- keys + values
        for k, v in self.atom_usage.items():
            metadata_bytes += 8 + 8  # int key + float value (approximate)
        breakdown['atom_metadata_bytes'] = metadata_bytes

        # --- Node metadata (count ints + state strings + residual floats) ---
        node_meta_bytes = 0.0
        for cls, clusters in self.class_clusters.items():
            for c in clusters:
                node_meta_bytes += 4   # count (int32 approx)
                node_meta_bytes += 8   # residual (float64)
                node_meta_bytes += len(c.node_state.encode('utf-8'))
        breakdown['node_metadata_bytes'] = node_meta_bytes

        # --- Edges ---
        edge_bytes = 0.0
        for cls, edge_dict in self.class_edges.items():
            for node_idx, neighbors in edge_dict.items():
                # Per directed edge entry: neighbor index + age; plus source key.
                edge_bytes += 4 + len(neighbors) * (4 + 4)
        breakdown['edges_bytes'] = edge_bytes

        edge_reliability_bytes = 0.0
        for cls, edge_dict in self.class_edge_reliability.items():
            for node_idx, neighbors in edge_dict.items():
                edge_reliability_bytes += 4 + len(neighbors) * (4 + 4)
        breakdown['edge_reliability_bytes'] = edge_reliability_bytes

        # --- Class-level NCM centers used by global-local readout ---
        class_mu_bytes = 0.0
        class_mu_raw_bytes = 0.0
        for mu in self.class_mu.values():
            if isinstance(mu, np.ndarray):
                class_mu_bytes += mu.nbytes
        for mu in self.class_mu_raw.values():
            if isinstance(mu, np.ndarray):
                class_mu_raw_bytes += mu.nbytes
        breakdown['class_mu_bytes'] = class_mu_bytes
        breakdown['class_mu_raw_bytes'] = class_mu_raw_bytes

        # --- Node center vectors (raw compact stores normalized centers; dict compact does not) ---
        node_center_norm_bytes = 0.0
        node_center_raw_bytes = 0.0
        for cls, clusters in self.class_clusters.items():
            for c in clusters:
                if isinstance(c.center, np.ndarray):
                    node_center_norm_bytes += c.center.nbytes
                if isinstance(c.center_raw, np.ndarray):
                    node_center_raw_bytes += c.center_raw.nbytes
        node_dict_recon_raw_bytes = 0.0
        for cls, clusters in self.class_clusters.items():
            for c in clusters:
                dict_recon_raw = getattr(c, 'dict_recon_raw', None)
                if isinstance(dict_recon_raw, np.ndarray):
                    node_dict_recon_raw_bytes += dict_recon_raw.nbytes
        node_center_bytes = node_center_norm_bytes + node_center_raw_bytes + node_dict_recon_raw_bytes
        breakdown['node_center_norm_bytes'] = node_center_norm_bytes
        breakdown['node_center_raw_bytes'] = node_center_raw_bytes
        breakdown['node_dict_recon_raw_bytes'] = node_dict_recon_raw_bytes
        breakdown['node_centers_bytes'] = node_center_bytes

        # --- Direction 1 raw fallback cache ---
        fallback_storage = self._raw_fallback_storage_bytes()
        raw_fallback_total_bytes = float(fallback_storage.get('raw_fallback_total_bytes', 0.0))
        breakdown.update(fallback_storage)
        fallback_gate_storage = self._raw_fallback_gate_model_bytes()
        raw_fallback_gate_model_bytes = float(
            fallback_gate_storage.get('raw_fallback_gate_model_bytes', 0.0)
        )
        breakdown.update(fallback_gate_storage)

        if self.use_dict_coding and atom_deployable_bytes > 0.0:
            # Deployable LifeTopoDict storage keeps one atom matrix, sparse
            # coefficients, lifecycle/topology metadata, and class NCM centers.
            compact = (
                atom_deployable_bytes +
                coeff_bytes +
                metadata_bytes +
                node_meta_bytes +
                edge_bytes +
                edge_reliability_bytes +
                class_mu_bytes
            )
        else:
            # Raw HC-SOINN deployable storage must include prototype centers;
            # otherwise raw/no-dict baselines are under-counted.
            compact = (
                class_mu_bytes +
                node_center_norm_bytes +
                node_meta_bytes +
                edge_bytes +
                edge_reliability_bytes
            )
        if getattr(self, 'fallback_memory_accounting', True):
            compact += raw_fallback_total_bytes + raw_fallback_gate_model_bytes
        breakdown['compact_deployable_bytes'] = compact

        # --- Caches (inference predict cache) ---
        cache_bytes = 0.0
        for k, v in self._predict_cache.items():
            cache_bytes += _array_nbytes(v)
        for k, v in getattr(self, '_raw_fallback_predict_cache', {}).items():
            cache_bytes += _array_nbytes(v)
        breakdown['caches_bytes'] = cache_bytes

        # --- Buffers ---
        buffers_bytes = 0.0
        for cls, bufs in self.buffers.items():
            for b in bufs:
                if isinstance(b, np.ndarray):
                    buffers_bytes += b.nbytes
        breakdown['buffers_bytes'] = buffers_bytes

        # --- Frozen clusters ---
        frozen_bytes = 0.0
        for cls, frozen in self.frozen_clusters.items():
            for c in frozen:
                if isinstance(c.center, np.ndarray):
                    frozen_bytes += c.center.nbytes
                if isinstance(c.center_raw, np.ndarray):
                    frozen_bytes += c.center_raw.nbytes
                dict_recon_raw = getattr(c, 'dict_recon_raw', None)
                if isinstance(dict_recon_raw, np.ndarray):
                    frozen_bytes += dict_recon_raw.nbytes
                if c.coeff is not None and isinstance(c.coeff, np.ndarray):
                    frozen_bytes += c.coeff.nbytes
        breakdown['frozen_bytes'] = frozen_bytes

        star_feature_bytes = (
            _array_nbytes(getattr(self, 'star_feature_anchors', None)) +
            _array_nbytes(getattr(self, 'feature_anchors', None))
        )
        star_image_bytes = (
            _array_nbytes(getattr(self, 'star_image_anchors', None)) +
            _array_nbytes(getattr(self, 'image_anchors', None))
        )
        breakdown['star_feature_anchors_bytes'] = star_feature_bytes
        breakdown['star_image_anchors_bytes'] = star_image_bytes

        actual = (
            atom_actual_bytes +
            coeff_bytes +
            metadata_bytes +
            node_meta_bytes +
            edge_bytes +
            edge_reliability_bytes +
            class_mu_bytes +
            class_mu_raw_bytes +
            node_center_bytes +
            raw_fallback_total_bytes +
            raw_fallback_gate_model_bytes +
            cache_bytes +
            buffers_bytes +
            frozen_bytes +
            star_feature_bytes +
            star_image_bytes
        )
        breakdown['actual_implementation_bytes'] = actual

        # --- Convert all _bytes keys to _mb ---
        MB = 1024.0 * 1024.0
        for k in list(breakdown.keys()):
            if k.endswith('_bytes'):
                mb_key = k.replace('_bytes', '_mb')
                breakdown[mb_key] = breakdown[k] / MB

        return breakdown

    def compute_diagnostics(self) -> Dict[str, object]:
        """F-6: Compute and return a comprehensive diagnostic report.

        Returns:
            Dictionary with the following keys:
            - ``PAD``: Prototype Alignment Degree (F-3)
            - ``EffRank``: Effective Dictionary Rank (F-4)
            - ``GTE``: Growth Trigger Effectiveness (F-5)
            - ``atom_count``: Current number of dictionary atoms
            - ``protected_ratio``: Fraction of atoms in 'protected' state
            - ``inactive_ratio``: Fraction of atoms in 'inactive' state
            - ``avg_residual``: Mean reconstruction residual across active nodes
        """
        M = len(self.atom_states) if self.atom_states else 0

        # PAD
        pad = self.compute_PAD()

        # EffRank
        eff_rank = self.compute_eff_rank()
        coeff_eff_rank = self.compute_coeff_eff_rank()

        # GTE
        gte = self.compute_GTE()

        # Protected / inactive ratio
        if M > 0:
            n_protected = sum(1 for s in self.atom_states if s == 'protected')
            n_inactive = sum(1 for s in self.atom_states if s == 'inactive')
            protected_ratio = float(n_protected) / float(M)
            inactive_ratio = float(n_inactive) / float(M)
        else:
            protected_ratio = 0.0
            inactive_ratio = 0.0

        # Average residual across active nodes
        residuals: List[float] = []
        for cls, clusters in self.class_clusters.items():
            for node in clusters:
                if node.node_state != 'inactive' and node.coeff is not None:
                    residuals.append(node.residual)
        avg_residual = float(np.mean(residuals)) if residuals else 0.0

        # Memory breakdown
        memory = self._compute_memory_breakdown()
        lifecycle_summary = dict(getattr(self, '_last_lifecycle_summary', {}))
        predict_cache_filter_stats = dict(getattr(self, '_last_predict_cache_filter_stats', {}))
        fallback_stats = self._summarize_raw_fallback_eval_stats()
        fallback_gate_stats = dict(getattr(self, '_raw_fallback_gate_fit_stats', {}))
        self._last_fallback_stats = fallback_stats
        inactive_node_ratio = float(lifecycle_summary.get('inactive_node_ratio', 0.0))
        inactive_nodes_filtered = float(predict_cache_filter_stats.get('inactive_nodes_filtered', 0.0))
        inactive_node_filter_ratio = float(
            predict_cache_filter_stats.get('inactive_node_filter_ratio', 0.0)
        )

        report: Dict[str, object] = {
            'PAD': pad,
            'EffRank': eff_rank,
            'coeff_eff_rank': coeff_eff_rank,
            'GTE': gte,
            'atom_count': M,
            'protected_ratio': protected_ratio,
            'inactive_ratio': inactive_ratio,
            'inactive_node_ratio': inactive_node_ratio,
            'inactive_nodes_filtered': inactive_nodes_filtered,
            'inactive_node_filter_ratio': inactive_node_filter_ratio,
            'avg_residual': avg_residual,
            'memory': memory,
            'lifecycle_summary': lifecycle_summary,
            'predict_cache_filter_stats': predict_cache_filter_stats,
            'edge_score_stats': dict(getattr(self, '_last_edge_score_stats', {})),
            'raw_fallback_stats': fallback_stats,
            'raw_fallback_gate_stats': fallback_gate_stats,
        }

        logging.info(
            f"[LifeTopoDict] Diagnostics: PAD={pad:.4f}, EffRank={eff_rank:.2f}, "
            f"coeff_eff_rank={coeff_eff_rank:.2f}, "
            f"GTE={gte:.4f}, atoms={M}, "
            f"protected={protected_ratio:.2%}, inactive={inactive_ratio:.2%}, "
            f"inactive_nodes={inactive_node_ratio:.2%}, "
            f"filtered_inactive_nodes={inactive_node_filter_ratio:.2%}, "
            f"avg_residual={avg_residual:.4f}"
        )

        return report

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def invalidate_cache(self) -> None:
        """Invalidate cached tensors used by predict_topk(). Safe to call often."""
        self._predict_cache_dirty = True
        self._predict_cache.clear()
        self._raw_fallback_predict_cache_dirty = True
        self._raw_fallback_predict_cache.clear()
        self._raw_fallback_gate_model_dirty = True
        self._raw_fallback_gate_model = {}

    # ------------------------------------------------------------------ #
    # Inference profiling helpers
    # ------------------------------------------------------------------ #
    def reset_profile_stats(self) -> None:
        self._profile_stats = {
            "calls": 0,
            "samples": 0,
            "total_sec": 0.0,
            "steps_sec": defaultdict(float),
        }

    def set_inference_profiling(
        self,
        enabled: bool,
        reset: bool = False,
        sync_cuda: Optional[bool] = None,
    ) -> None:
        self.enable_inference_profiling = bool(enabled)
        if sync_cuda is not None:
            self.profile_sync_cuda = bool(sync_cuda)
        if reset:
            self.reset_profile_stats()

    def get_profile_stats(self, reset: bool = False) -> Dict[str, object]:
        calls = int(self._profile_stats.get("calls", 0))
        samples = int(self._profile_stats.get("samples", 0))
        total_sec = float(self._profile_stats.get("total_sec", 0.0))
        steps_raw = self._profile_stats.get("steps_sec", {})
        steps_sec = {k: float(v) for k, v in sorted(steps_raw.items(), key=lambda x: x[1], reverse=True)}
        out = {
            "calls": calls,
            "samples": samples,
            "total_sec": total_sec,
            "avg_ms_per_call": (total_sec / calls * 1000.0) if calls > 0 else 0.0,
            "avg_ms_per_sample": (total_sec / samples * 1000.0) if samples > 0 else 0.0,
            "steps_sec": steps_sec,
            "steps_ratio": {k: (v / total_sec if total_sec > 0 else 0.0) for k, v in steps_sec.items()},
        }
        if reset:
            self.reset_profile_stats()
        return out

    def _profile_sync(self, device: torch.device) -> None:
        if (
            self.enable_inference_profiling
            and self.profile_sync_cuda
            and isinstance(device, torch.device)
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(device=device)

    def _profile_tic(self, device: torch.device) -> float:
        self._profile_sync(device)
        return time.perf_counter()

    def _profile_toc(self, step_name: str, t0: float, device: torch.device) -> float:
        self._profile_sync(device)
        dt = time.perf_counter() - t0
        self._profile_stats["steps_sec"][step_name] += dt
        return dt

    def _ensure_predict_cache(
        self,
        device: torch.device,
        query_dim: int,
        valid_classes: List[int],
    ) -> None:
        """
        Build/cache prototype tensors on the target device to avoid per-call numpy concat and H2D copies.
        Cache key depends on: device, query_dim, and the ordered valid_classes list.
        """
        device_key = str(device)
        classes_key = tuple(int(c) for c in valid_classes)

        if (
            (not self._predict_cache_dirty)
            and self._predict_cache.get("device_key") == device_key
            and self._predict_cache.get("query_dim") == int(query_dim)
            and self._predict_cache.get("classes_key") == classes_key
        ):
            return

        # ---------- NCM centers (already normalized) ----------
        ncm_centers_np = np.stack([self.class_mu[cls] for cls in valid_classes]).astype(np.float32, copy=False)
        ncm_centers_t = torch.from_numpy(ncm_centers_np).to(device=device, dtype=torch.float32)

        # ---------- Prototypes (sub-clusters; fallback to NCM if empty) ----------
        all_protos: List[np.ndarray] = []
        proto_labels: List[int] = []       # original class id per proto
        proto_class_index: List[int] = []  # index in valid_classes per proto (0..C-1)
        proto_node_index: List[int] = []   # local node index inside class; -1 for NCM fallback
        proto_lookup: Dict[Tuple[int, int], int] = {}
        cache_total_nodes = 0
        cache_inactive_nodes_filtered = 0

        for class_index, cls in enumerate(valid_classes):
            clusters = self.class_clusters.get(cls, [])
            if clusters:
                cluster_dims = [c.center.shape[0] for c in clusters]
                if all(dim == query_dim for dim in cluster_dims):
                    cache_total_nodes += len(clusters)
                    # E-6: Filter out inactive nodes when dict coding is enabled
                    if getattr(self, 'use_dict_coding', False):
                        cls_node_states = self.node_states.get(cls, [])
                        active_clusters = []
                        active_node_indices = []
                        for ci, c in enumerate(clusters):
                            state = cls_node_states[ci] if ci < len(cls_node_states) else 'plastic'
                            if state != 'inactive':
                                active_clusters.append(c)
                                active_node_indices.append(ci)
                            else:
                                cache_inactive_nodes_filtered += 1
                        # If all nodes are inactive, fall through to NCM fallback
                        if active_clusters:
                            cls_protos = np.stack([c.center for c in active_clusters]).astype(np.float32, copy=False)
                            start_pos = len(proto_labels)
                            all_protos.append(cls_protos)
                            n_p = int(cls_protos.shape[0])
                            proto_labels.extend([int(cls)] * n_p)
                            proto_class_index.extend([int(class_index)] * n_p)
                            proto_node_index.extend([int(i) for i in active_node_indices])
                            for offset, node_idx in enumerate(active_node_indices):
                                proto_lookup[(int(cls), int(node_idx))] = start_pos + offset
                            continue
                    else:
                        cls_protos = np.stack([c.center for c in clusters]).astype(np.float32, copy=False)
                        start_pos = len(proto_labels)
                        all_protos.append(cls_protos)
                        n_p = int(cls_protos.shape[0])
                        proto_labels.extend([int(cls)] * n_p)
                        proto_class_index.extend([int(class_index)] * n_p)
                        proto_node_index.extend([int(i) for i in range(len(clusters))])
                        for node_idx in range(len(clusters)):
                            proto_lookup[(int(cls), int(node_idx))] = start_pos + node_idx
                        continue

            # Fallback: at least one proto per class (use NCM center)
            mu = self.class_mu[cls].astype(np.float32, copy=False)
            all_protos.append(mu[np.newaxis, :])
            proto_labels.append(int(cls))
            proto_class_index.append(int(class_index))
            proto_node_index.append(-1)

        all_protos_np = np.concatenate(all_protos, axis=0).astype(np.float32, copy=False)
        all_protos_t = torch.from_numpy(all_protos_np).to(device=device, dtype=torch.float32)
        proto_labels_t = torch.tensor(proto_labels, device=device, dtype=torch.long)
        proto_class_index_t = torch.tensor(proto_class_index, device=device, dtype=torch.long)
        proto_node_index_t = torch.tensor(proto_node_index, device=device, dtype=torch.long)

        # Prototypes and NCM centers are expected to already be L2-normalized.
        # We keep them as-is to avoid extra normalize() cost per call.
        self._predict_cache = {
            "device_key": device_key,
            "query_dim": int(query_dim),
            "classes_key": classes_key,
            "ncm_centers_t": ncm_centers_t,               # [C, D]
            "all_protos_t": all_protos_t,                 # [M, D]
            "proto_labels_t": proto_labels_t,             # [M]
            "proto_class_index_t": proto_class_index_t,   # [M] in 0..C-1
            "proto_node_index_t": proto_node_index_t,     # [M] local node index or -1
            "proto_lookup": proto_lookup,
        }
        self._last_predict_cache_filter_stats = {
            "total_nodes": float(cache_total_nodes),
            "inactive_nodes_filtered": float(cache_inactive_nodes_filtered),
            "inactive_node_filter_ratio": (
                float(cache_inactive_nodes_filtered) / float(cache_total_nodes)
                if cache_total_nodes > 0 else 0.0
            ),
        }
        self._predict_cache_dirty = False

    # ------------------------------------------------------------------ #
    # STAR: Force Freeze Methods
    # ------------------------------------------------------------------ #
    def freeze_nodes(self, cls: int) -> None:
        """Force backup current nodes state."""
        if cls in self.class_clusters:
            # Deep copy to ensure independence
            self.frozen_clusters[cls] = [
                _Cluster(
                    center=c.center.copy(),
                    count=c.count,
                    center_raw=c.center_raw.copy() if c.center_raw is not None else None,
                    coeff=c.coeff.copy() if c.coeff is not None else None,
                    residual=c.residual,
                    node_state=c.node_state,
                    dict_recon_raw=(
                        c.dict_recon_raw.copy()
                        if getattr(c, 'dict_recon_raw', None) is not None else None
                    )
                )
                for c in self.class_clusters[cls]
            ]
            logging.info(f"[HC-SOINN] Class {cls} nodes FROZEN.")

    def restore_frozen_nodes(self, cls: int) -> None:
        """Force restore nodes state from frozen backup."""
        if cls in self.frozen_clusters:
            self.class_clusters[cls] = []
            for c in self.frozen_clusters[cls]:
                # Deep copy raw center
                center_raw = c.center_raw.copy() if c.center_raw is not None else None

                if self.use_dict_coding and c.coeff is not None:
                    center = c.center.copy()
                elif center_raw is not None:
                    norm = np.linalg.norm(center_raw)
                    if norm > 1e-9:
                        center = center_raw / norm
                    else:
                        center = c.center.copy() # Fallback
                else:
                    center = c.center.copy()
                    
                self.class_clusters[cls].append(_Cluster(
                    center, c.count, center_raw,
                    coeff=c.coeff.copy() if c.coeff is not None else None,
                    residual=c.residual,
                    node_state=c.node_state,
                    dict_recon_raw=(
                        c.dict_recon_raw.copy()
                        if getattr(c, 'dict_recon_raw', None) is not None else None
                    )
                ))
                
            logging.info(f"[HC-SOINN] Class {cls} nodes RESTORED from frozen state (Normalized re-calculated).")

    def _compute_edge_score_adjustment(
        self,
        dist_proto_all_t: torch.Tensor,
        valid_classes: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Return an optional edge-aware distance adjustment for P1 ablation.

        Negative values improve a class distance when the nearest node has
        reliable, query-supported neighbors; positive values penalize old edges.
        """
        N = int(dist_proto_all_t.shape[0])
        C = len(valid_classes)
        adjustment = np.zeros((N, C), dtype=np.float32)

        if not getattr(self, 'use_edge_aware_scoring', False):
            self._last_edge_score_stats = {
                'edge_use_rate': 0.0,
                'edge_class_use_rate': 0.0,
                'edge_margin_contribution': 0.0,
                'edge_risk_penalty': 0.0,
            }
            return torch.from_numpy(adjustment).to(device=device, dtype=dist_proto_all_t.dtype)

        proto_class_index_t = self._predict_cache.get("proto_class_index_t")
        proto_node_index_t = self._predict_cache.get("proto_node_index_t")
        proto_lookup = self._predict_cache.get("proto_lookup", {})
        if proto_class_index_t is None or proto_node_index_t is None:
            return torch.from_numpy(adjustment).to(device=device, dtype=dist_proto_all_t.dtype)

        dist_np = dist_proto_all_t.detach().cpu().numpy()
        proto_class_index = proto_class_index_t.detach().cpu().numpy()
        proto_node_index = proto_node_index_t.detach().cpu().numpy()

        sample_used = np.zeros(N, dtype=bool)
        class_use_count = 0
        support_values: List[float] = []
        risk_values: List[float] = []
        adjustment_values: List[float] = []

        gamma = float(getattr(self, 'edge_score_gamma', 0.1))
        eta = float(getattr(self, 'edge_score_eta', 0.05))

        for class_index, cls in enumerate(valid_classes):
            proto_positions = np.where(proto_class_index == class_index)[0]
            if proto_positions.size == 0:
                continue

            edges_for_class = self.class_edges.get(int(cls), {})
            reliab_for_class = self.class_edge_reliability.get(int(cls), {})
            if not edges_for_class:
                continue

            nearest_local = np.argmin(dist_np[:, proto_positions], axis=1)
            nearest_proto_positions = proto_positions[nearest_local]
            nearest_node_indices = proto_node_index[nearest_proto_positions]

            for row, node_idx_raw in enumerate(nearest_node_indices):
                node_idx = int(node_idx_raw)
                if node_idx < 0:
                    continue
                neighbors = edges_for_class.get(node_idx, {})
                if not neighbors:
                    continue

                support_terms: List[float] = []
                risk_terms: List[float] = []
                for nbr_raw, age_raw in neighbors.items():
                    nbr = int(nbr_raw)
                    proto_pos = proto_lookup.get((int(cls), nbr))
                    if proto_pos is None:
                        continue
                    age = float(age_raw)
                    neighbor_sim = max(0.0, 1.0 - float(dist_np[row, proto_pos]))
                    reliability = float(reliab_for_class.get(node_idx, {}).get(nbr, 0.0))
                    support_terms.append(neighbor_sim * reliability)
                    risk_terms.append(age / (age + 1.0))

                if not support_terms:
                    continue

                edge_support = float(np.mean(support_terms))
                edge_risk = float(np.mean(risk_terms)) if risk_terms else 0.0
                delta = eta * edge_risk - gamma * edge_support
                adjustment[row, class_index] = delta
                sample_used[row] = True
                class_use_count += 1
                support_values.append(edge_support)
                risk_values.append(edge_risk)
                adjustment_values.append(delta)

        denom = float(max(N * C, 1))
        self._last_edge_score_stats = {
            'edge_use_rate': float(sample_used.mean()) if N > 0 else 0.0,
            'edge_class_use_rate': float(class_use_count / denom),
            'edge_margin_contribution': float(np.mean(support_values)) if support_values else 0.0,
            'edge_risk_penalty': float(np.mean(risk_values)) if risk_values else 0.0,
            'edge_score_adjustment': float(np.mean(adjustment_values)) if adjustment_values else 0.0,
        }

        return torch.from_numpy(adjustment).to(device=device, dtype=dist_proto_all_t.dtype)

    # ------------------------------------------------------------------ #
    # DEBUG: Snapshot and comparison methods
    # ------------------------------------------------------------------ #
    def save_cluster_snapshot(self, task_id: int, class_list: Optional[List[int]] = None) -> None:
        """
        Save a snapshot of cluster centers for debugging.
        
        Args:
            task_id: Current task ID
            class_list: List of classes to save (None = all classes)
        """
        if task_id not in self.cluster_snapshots:
            self.cluster_snapshots[task_id] = {}
        
        classes_to_save = class_list if class_list is not None else list(self.class_clusters.keys())
        
        for cls in classes_to_save:
            if cls in self.class_clusters and len(self.class_clusters[cls]) > 0:
                clusters = self.class_clusters[cls]
                centers_normalized = np.array([c.center.copy() for c in clusters])
                centers_raw = np.array([
                    (c.center_raw.copy() if isinstance(c.center_raw, np.ndarray) else c.center.copy())
                    for c in clusters
                ])
                self.cluster_snapshots[task_id][cls] = (centers_normalized, centers_raw)
                
    
    def compare_cluster_snapshots(self, task_id1: int, task_id2: int, class_list: Optional[List[int]] = None) -> None:
        """
        Compare cluster snapshots between two tasks.
        
        Args:
            task_id1: First task ID
            task_id2: Second task ID
            class_list: List of classes to compare (None = all common classes)
        """
        if task_id1 not in self.cluster_snapshots or task_id2 not in self.cluster_snapshots:
            return
        
        snapshot1 = self.cluster_snapshots[task_id1]
        snapshot2 = self.cluster_snapshots[task_id2]
        
        classes_to_compare = class_list if class_list is not None else list(set(snapshot1.keys()) & set(snapshot2.keys()))
        
        # Debug logging removed for performance
        for cls in sorted(classes_to_compare):
            if cls not in snapshot1 or cls not in snapshot2:
                continue
            
            centers_norm1, centers_raw1 = snapshot1[cls]
            centers_norm2, centers_raw2 = snapshot2[cls]
            
            # Check if dimensions match
            if centers_norm1.shape != centers_norm2.shape:
                logging.warning(
                    f"[DEBUG] Class {cls}: Shape mismatch! "
                    f"Task {task_id1}: {centers_norm1.shape}, Task {task_id2}: {centers_norm2.shape}"
                )
                continue
            
            # Compute differences (for internal use only, no logging)
            diff_norm = np.abs(centers_norm1 - centers_norm2)
            diff_raw = np.abs(centers_raw1 - centers_raw2)
    
    # ------------------------------------------------------------------ #
    # Lifecycle methods (Agent E: E-2, E-3, E-4)
    # ------------------------------------------------------------------ #
    def _compute_atom_usage(self) -> Dict[int, float]:
        """E-2: Return per-atom EMA usage rate.

        Returns ``self.atom_usage_ema`` directly, which is maintained by
        ``_encode_nodes_with_dict()`` via per-task counting + EMA blending.

        If ``atom_usage_ema`` is empty (first call before any encoding), a
        fallback computation is performed based on stored coefficients.

        Returns:
            Dict mapping atom index -> EMA usage rate.
        """
        if not getattr(self, 'use_dict_coding', False):
            return {}

        dict_atoms_hat = getattr(self, 'dict_atoms_hat', None)
        if dict_atoms_hat is None or dict_atoms_hat.shape[0] == 0:
            return {}

        M = dict_atoms_hat.shape[0]

        # Fast path: return EMA directly if available
        ema = getattr(self, 'atom_usage_ema', {})
        if ema:
            # Ensure all atoms are present (new atoms may have been added)
            for m in range(M):
                if m not in ema:
                    ema[m] = 0.0
            return ema

        # Fallback: compute from stored coefficients (first-time / cold start)
        atom_hit_count = np.zeros(M, dtype=np.float64)
        total_selections = 0
        k = min(self.dict_sparse_k, M)

        for cls, clusters in self.class_clusters.items():
            for cluster in clusters:
                coeff = getattr(cluster, 'coeff', None)
                if coeff is not None:
                    topk_indices = self._topk_nonzero_coeff_indices(coeff, M, k)
                    for idx in topk_indices:
                        atom_hit_count[int(idx)] += 1.0
                    total_selections += int(len(topk_indices))

        usage = {}
        if total_selections > 0:
            for m in range(M):
                usage[m] = float(atom_hit_count[m] / total_selections)
        else:
            usage = {m: 0.0 for m in range(M)}

        # Seed the EMA with the fallback computation
        self.atom_usage_ema = dict(usage)
        return usage

    def _compute_old_support(self) -> Dict[int, float]:
        """E-3: Compute per-atom average old-class support strength.

        For each atom m, compute the mean |a_{c,m}| across all nodes of
        old classes.  Normalise by the number of old-class nodes (not by
        the number of old classes) to avoid monotonic growth.

        Returns:
            Dict mapping atom index -> normalised old support.
        """
        if not getattr(self, 'use_dict_coding', False):
            return {}

        dict_atoms_hat = getattr(self, 'dict_atoms_hat', None)
        if dict_atoms_hat is None or dict_atoms_hat.shape[0] == 0:
            return {}

        M = dict_atoms_hat.shape[0]
        atom_abs_sum = np.zeros(M, dtype=np.float64)
        node_count = 0

        for cls, clusters in self.class_clusters.items():
            if not self._is_lifecycle_old_class(cls):
                continue
            for cluster in clusters:
                coeff = getattr(cluster, 'coeff', None)
                if coeff is not None:
                    coeff_arr = np.abs(np.asarray(coeff).ravel())
                    if coeff_arr.size < M:
                        padded = np.zeros(M, dtype=np.float64)
                        padded[:coeff_arr.size] = coeff_arr
                        coeff_arr = padded
                    else:
                        coeff_arr = coeff_arr[:M]
                    atom_abs_sum += coeff_arr
                    node_count += 1

        if node_count == 0:
            return {m: 0.0 for m in range(M)}

        # Normalise by number of nodes (not classes) to avoid monotonic growth
        mean_support = atom_abs_sum / float(node_count)

        # Further normalise to [0, 1] range by dividing by max
        max_support = mean_support.max()
        if max_support > 1e-8:
            mean_support = mean_support / max_support

        return {m: float(mean_support[m]) for m in range(M)}

    def _is_lifecycle_old_class(self, cls: int) -> bool:
        """Return True when a class belongs to the old-class side of this task."""
        known_classes = getattr(self, 'known_classes_before_task', None)
        return known_classes is None or int(cls) < int(known_classes)

    @staticmethod
    def _topk_nonzero_coeff_indices(
        coeff: np.ndarray,
        M: int,
        k: int,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Return non-zero top-k coefficient indices, bounded to current atom count."""
        coeff_arr = np.abs(np.asarray(coeff).ravel())
        if M <= 0 or k <= 0 or coeff_arr.size == 0:
            return np.empty(0, dtype=np.int64)
        coeff_arr = coeff_arr[:M]
        positive = np.flatnonzero(coeff_arr > eps)
        if positive.size == 0:
            return np.empty(0, dtype=np.int64)
        if positive.size <= k:
            return positive.astype(np.int64, copy=False)
        local = np.argpartition(-coeff_arr[positive], kth=k - 1)[:k]
        return positive[local].astype(np.int64, copy=False)

    def _collect_old_topk_atoms(self, M: int, k: int) -> Set[int]:
        """Collect atoms used by old-class nodes' current sparse top-k codes."""
        old_topk_atoms: Set[int] = set()
        for cls, clusters in self.class_clusters.items():
            if not self._is_lifecycle_old_class(cls):
                continue
            for cluster in clusters:
                coeff = getattr(cluster, 'coeff', None)
                if coeff is None:
                    continue
                topk_idx = self._topk_nonzero_coeff_indices(coeff, M, k)
                for idx in topk_idx:
                    old_topk_atoms.add(int(idx))
        return old_topk_atoms

    def _update_lifecycle_states(self) -> None:
        """E-4: Update three-state lifecycle: protected / plastic / inactive.

        State transitions:
        - protected: old_support[m] > theta_support (atom is heavily used by old classes)
        - inactive: usage[m] < theta_usage for T_inactive consecutive tasks
                    AND old_support[m] < theta_support
        - plastic: default state (neither protected nor inactive)
        """
        if not getattr(self, 'use_dict_coding', False):
            return

        dict_atoms = getattr(self, 'dict_atoms', None)
        if dict_atoms is None:
            return

        M = dict_atoms.shape[0]

        # P0-4 ablation gate: keep all atoms plastic when lifecycle is disabled
        if not self.use_lifecycle:
            self.atom_states = ['plastic'] * M
            self.inactive_mask = np.ones(M, dtype=np.bool_)
            total_nodes = 0
            for cls, clusters in self.class_clusters.items():
                self.node_states[cls] = ['plastic'] * len(clusters)
                total_nodes += len(clusters)
                for cluster in clusters:
                    cluster.node_state = 'plastic'
            self._last_lifecycle_summary = {
                'protected_nodes': 0.0,
                'plastic_nodes': float(total_nodes),
                'inactive_nodes': 0.0,
                'inactive_node_ratio': 0.0,
                'node_inactive_threshold': float(self.lifecycle_node_inactive_threshold),
            }
            return

        k = min(self.dict_sparse_k, M)

        # 1. Compute usage and old support
        usage = self._compute_atom_usage()
        old_support = self._compute_old_support()
        self.old_support = dict(old_support)
        old_topk_atoms = (
            self._collect_old_topk_atoms(M, k)
            if self.lifecycle_protect_old_topk else set()
        )

        # 2. Update atom_states (list maintained by Agent D)
        atom_states = getattr(self, 'atom_states', None)
        if atom_states is None:
            self.atom_states = ['plastic'] * M
            atom_states = self.atom_states

        # Ensure atom_states length matches current M
        while len(atom_states) < M:
            atom_states.append('plastic')
        while len(atom_states) > M:
            atom_states.pop()

        # 3. Track consecutive low-usage streaks
        low_usage_streak = getattr(self, '_low_usage_streak', {})

        for m in range(M):
            u = usage.get(m, 0.0)
            s = old_support.get(m, 0.0)

            if u < self.lifecycle_theta_usage:
                low_usage_streak[m] = low_usage_streak.get(m, 0) + 1
            else:
                low_usage_streak[m] = 0

            # Determine new state
            if s > self.lifecycle_theta_support:
                new_state = 'protected'
            elif (low_usage_streak.get(m, 0) >= self.lifecycle_T_inactive
                  and s < self.lifecycle_theta_support):
                new_state = 'inactive'
            else:
                new_state = 'plastic'

            atom_states[m] = new_state

        inactive_before_safety = sum(1 for s in atom_states if s == 'inactive')

        old_topk_forced_protected = 0
        old_topk_reactivated = 0
        if old_topk_atoms:
            for m in old_topk_atoms:
                if m < 0 or m >= M:
                    continue
                if atom_states[m] == 'inactive':
                    old_topk_reactivated += 1
                if atom_states[m] != 'protected':
                    old_topk_forced_protected += 1
                atom_states[m] = 'protected'
                low_usage_streak[m] = 0

        alive_floor_reactivated = 0
        min_alive_ratio = min(1.0, max(0.0, self.lifecycle_min_alive_ratio))
        if M > 0 and min_alive_ratio > 0.0:
            required_alive = int(np.ceil(min_alive_ratio * float(M)))
            current_alive = M - sum(1 for s in atom_states if s == 'inactive')
            if current_alive < required_alive:
                need = required_alive - current_alive
                inactive_candidates = [m for m, s in enumerate(atom_states) if s == 'inactive']
                inactive_candidates.sort(
                    key=lambda m: (
                        old_support.get(m, 0.0),
                        usage.get(m, 0.0),
                        self.atom_usage_task.get(m, 0.0),
                        self.atom_usage.get(m, 0.0),
                        -m,
                    ),
                    reverse=True,
                )
                for m in inactive_candidates[:need]:
                    atom_states[m] = 'plastic'
                    low_usage_streak[m] = 0
                    alive_floor_reactivated += 1

        self._low_usage_streak = low_usage_streak

        # 4. Update inactive_mask
        self.inactive_mask = np.array(
            [atom_states[m] != 'inactive' for m in range(M)],
            dtype=np.bool_,
        )

        # 5. Update per-node node_state based on which atoms the node uses
        node_state_counts = {'protected': 0, 'plastic': 0, 'inactive': 0}
        node_inactive_threshold = min(1.0, max(0.0, float(self.lifecycle_node_inactive_threshold)))
        for cls, clusters in self.class_clusters.items():
            if cls not in self.node_states:
                self.node_states[cls] = []
            # Ensure node_states[cls] length matches clusters
            while len(self.node_states[cls]) < len(clusters):
                self.node_states[cls].append('plastic')
            self.node_states[cls] = self.node_states[cls][:len(clusters)]

            for i, cluster in enumerate(clusters):
                # A node is protected if ANY of its top-k atoms is protected
                # A node is inactive if its inactive top-k fraction reaches
                # lifecycle_node_inactive_threshold.  The default threshold
                # 1.0 preserves the previous "all inactive" semantics.
                # Otherwise plastic
                coeff = getattr(cluster, 'coeff', None)
                if coeff is not None:
                    topk_idx = self._topk_nonzero_coeff_indices(coeff, M, k)
                    if topk_idx.size == 0:
                        self.node_states[cls][i] = 'plastic'
                        cluster.node_state = 'plastic'
                        node_state_counts['plastic'] += 1
                        continue
                    topk_states = [atom_states[int(j)] for j in topk_idx]
                    inactive_fraction = (
                        sum(1 for st in topk_states if st == 'inactive')
                        / float(len(topk_states))
                    )

                    if any(st == 'protected' for st in topk_states):
                        self.node_states[cls][i] = 'protected'
                        cluster.node_state = 'protected'
                        node_state_counts['protected'] += 1
                    elif inactive_fraction >= node_inactive_threshold:
                        self.node_states[cls][i] = 'inactive'
                        cluster.node_state = 'inactive'
                        node_state_counts['inactive'] += 1
                    else:
                        self.node_states[cls][i] = 'plastic'
                        cluster.node_state = 'plastic'
                        node_state_counts['plastic'] += 1
                else:
                    self.node_states[cls][i] = 'plastic'
                    cluster.node_state = 'plastic'
                    node_state_counts['plastic'] += 1

        # 6. Update atom_old_tasks: which old-class nodes use each atom
        self.atom_old_tasks = {m: set() for m in range(M)}
        for cls, clusters in self.class_clusters.items():
            if not self._is_lifecycle_old_class(cls):
                continue
            for cluster in clusters:
                coeff = getattr(cluster, 'coeff', None)
                if coeff is not None:
                    topk_idx = self._topk_nonzero_coeff_indices(coeff, M, k)
                    for j in topk_idx:
                        self.atom_old_tasks[int(j)].add(cls)

        inactive_after_safety = sum(1 for s in atom_states if s == 'inactive')
        total_nodes = sum(node_state_counts.values())
        inactive_node_ratio = (
            float(node_state_counts['inactive']) / float(total_nodes)
            if total_nodes > 0 else 0.0
        )
        self._last_lifecycle_summary = {
            'protected_nodes': float(node_state_counts['protected']),
            'plastic_nodes': float(node_state_counts['plastic']),
            'inactive_nodes': float(node_state_counts['inactive']),
            'inactive_node_ratio': inactive_node_ratio,
            'node_inactive_threshold': float(node_inactive_threshold),
        }
        logging.info(
            f"[Lifecycle] Safety: old_topk_atoms={len(old_topk_atoms)}, "
            f"old_topk_forced_protected={old_topk_forced_protected}, "
            f"old_topk_reactivated={old_topk_reactivated}, "
            f"alive_floor_reactivated={alive_floor_reactivated}, "
            f"inactive_before={inactive_before_safety / max(M, 1):.2%}, "
            f"inactive_after={inactive_after_safety / max(M, 1):.2%}, "
            f"min_alive={min_alive_ratio:.2%}"
        )

        logging.info(
            f"[Lifecycle] Atom states: "
            f"protected={sum(1 for s in atom_states if s == 'protected')}, "
            f"plastic={sum(1 for s in atom_states if s == 'plastic')}, "
            f"inactive={sum(1 for s in atom_states if s == 'inactive')}"
        )
        logging.info(
            f"[Lifecycle] Node states: protected={node_state_counts['protected']}, "
            f"plastic={node_state_counts['plastic']}, "
            f"inactive={node_state_counts['inactive']}, "
            f"inactive_ratio={inactive_node_ratio:.2%}, "
            f"inactive_threshold={node_inactive_threshold:.2f}"
        )

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def add_features(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Handle add features."""
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if features.shape[0] == 0:
            return
        
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_feats = features[cls_mask]
            if cls_feats.shape[0] == 0:
                continue

            if cls not in self.buffers:
                self.buffers[cls] = []
            self.buffers[cls].append(cls_feats)

            cls_count_old = self.class_count.get(cls, 0)
            if cls in self.class_mu_raw:
                cls_sum_old = self.class_mu_raw[cls] * cls_count_old
            else:
                cls_sum_old = np.zeros(cls_feats.shape[1], dtype=np.float32)
            cls_sum_new = cls_sum_old + cls_feats.sum(axis=0)
            cls_count_new = cls_count_old + cls_feats.shape[0]
            self.class_count[cls] = cls_count_new
            # Store both raw and normalized versions
            cls_mean_raw = cls_sum_new / float(max(cls_count_new, 1))
            self.class_mu_raw[cls] = cls_mean_raw
            self.class_mu[cls] = _normalize(cls_mean_raw)
            
            # STAR: Save original NCM centers (only once, when first created)
            if cls not in self.class_mu_original:
                self.class_mu_raw_original[cls] = cls_mean_raw.copy()
                self.class_mu_original[cls] = _normalize(cls_mean_raw)

        # class_mu / buffers updated => invalidate inference cache
        self.invalidate_cache()

    def compress(self) -> None:
        """Handle compress."""
        # E-5: Build per-class protected node index sets from lifecycle state.
        # After compress() the cluster list is rebuilt by hierarchical clustering +
        # SOINN refinement, so old node indices no longer map to new ones.
        # Strategy: match old protected node centres to new cluster centres via
        # cosine similarity greedy matching, and only propagate protected status
        # for matches above a threshold.
        def _get_protected_indices(cls: int, old_clusters: Optional[List[_Cluster]] = None) -> Optional[Set[int]]:
            """Get indices of protected nodes for a class in the NEW cluster list."""
            cls_states = self.node_states.get(cls, [])
            if not cls_states or old_clusters is None:
                return None

            # Collect centres of old protected nodes
            protected_old_centres: List[np.ndarray] = []
            for idx, state in enumerate(cls_states):
                if idx < len(old_clusters) and state == 'protected':
                    protected_old_centres.append(old_clusters[idx].center)

            if not protected_old_centres:
                return None

            # Match against current (new) cluster centres
            new_clusters = self.class_clusters.get(cls, [])
            if not new_clusters:
                return None

            new_centres = np.stack([c.center for c in new_clusters], axis=0)  # [N_new, d]
            matched = set()
            used_new = set()
            for old_c in protected_old_centres:
                sims = new_centres @ old_c  # cosine sim (both normalised)
                # Mask already-matched indices
                for u in used_new:
                    sims[u] = -2.0
                best = int(np.argmax(sims))
                if sims[best] > 0.7:  # similarity threshold
                    matched.add(best)
                    used_new.add(best)

            return matched if matched else None

        def _get_protected_centers(cls: int, old_clusters: Optional[List[_Cluster]] = None) -> Optional[List[np.ndarray]]:
            """Collect old protected node centers for worker-side deletion/update gates."""
            cls_states = self.node_states.get(cls, [])
            if not cls_states or old_clusters is None:
                return None

            protected_centers: List[np.ndarray] = []
            for idx, old_cluster in enumerate(old_clusters):
                state = cls_states[idx] if idx < len(cls_states) else getattr(old_cluster, 'node_state', 'plastic')
                if state == 'protected':
                    protected_centers.append(old_cluster.center.copy())

            return protected_centers if protected_centers else None

        # Snapshot old clusters before compress for protected-index matching
        old_clusters_snapshot: Dict[int, List[_Cluster]] = {}
        for cls in list(self.buffers.keys()):
            if cls in self.class_clusters:
                old_clusters_snapshot[cls] = self.class_clusters[cls]

        tasks = []
        task_classes = []
        for cls, chunk_list in list(self.buffers.items()):
            if len(chunk_list) == 0:
                continue

            feats = np.concatenate(chunk_list, axis=0).astype(np.float32, copy=False)

            if cls in self.class_clusters and len(self.class_clusters[cls]) > 0:
                old_center_targets = [
                    c.center_raw if isinstance(c.center_raw, np.ndarray) else c.center
                    for c in self.class_clusters[cls]
                ]
                old_centers_raw = np.stack(old_center_targets, axis=0)
                feats = np.concatenate([feats, old_centers_raw], axis=0)

            target_k = feats.shape[0] if self.max_prototypes_per_class is None else min(
                self.max_prototypes_per_class, feats.shape[0]
            )

            protected_centers = None
            if self.use_dict_coding and self.use_protected_gate:
                protected_centers = _get_protected_centers(cls, old_clusters_snapshot.get(cls))

            tasks.append((
                cls, feats, target_k, self.tau_merge,
                self.linkage_method, self.distance_metric, self.max_prototypes_per_class,
                self.use_soinn_refinement, self.soinn_ad, self.soinn_lam,
                self.soinn_threshold_scale, self.soinn_max_iter, self.soinn_max_degree_for_removal,
                protected_centers
            ))
            task_classes.append(cls)

        if tasks:
            logging.info(f"[HC-SOINN] Compressing {len(tasks)} classes using multiprocessing...")
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(_compress_class_worker, tasks))
            
            for cls, clusters, hierarchical_count, final_count, soinn_edges in results:

                self.class_clusters[cls] = clusters

                # P0-2: Remap protected status from old clusters to new clusters
                old_cls_clusters = old_clusters_snapshot.get(cls)
                if (old_cls_clusters is not None and self.use_dict_coding
                        and self.use_protected_gate):
                    matched_protected = _get_protected_indices(cls, old_cls_clusters)
                    if matched_protected:
                        # Update node_states for this class and sync to _Cluster.node_state
                        if cls not in self.node_states:
                            self.node_states[cls] = ['plastic'] * len(clusters)
                        else:
                            self.node_states[cls] = ['plastic'] * len(clusters)
                        for idx in matched_protected:
                            if idx < len(clusters):
                                self.node_states[cls][idx] = 'protected'
                                clusters[idx].node_state = 'protected'
                        logging.info(
                            f"[LifeTopoDict] Protected remap for class {cls}: "
                            f"{len(matched_protected)}/{len(old_cls_clusters)} old protected -> "
                            f"{len(matched_protected)} new matched."
                        )
                
                # STAR: Save original clusters (only once, when first created)
                if cls not in self.class_clusters_original:
                    # Deep copy clusters to preserve original state
                    self.class_clusters_original[cls] = [
                        _Cluster(
                            center=c.center.copy(),
                            count=c.count,
                            center_raw=c.center_raw.copy() if c.center_raw is not None else None,
                            dict_recon_raw=(
                                c.dict_recon_raw.copy()
                                if getattr(c, 'dict_recon_raw', None) is not None else None
                            )
                        )
                        for c in clusters
                    ]
                
                # DEBUG: Log new cluster statistics after replacement
                if len(clusters) > 0:
                    new_centers = np.array([c.center for c in clusters])
                    new_centers_raw = np.array([
                        c.center_raw if isinstance(c.center_raw, np.ndarray) else c.center
                        for c in clusters
                    ])
                    new_mean_norm = np.mean(np.linalg.norm(new_centers, axis=1))
                    new_mean_norm_raw = np.mean(np.linalg.norm(new_centers_raw, axis=1))
                    logging.info(
                        f"[HC-SOINN] Class {cls}: Created {len(clusters)} new nodes "
                        # f"(center_mean_norm={new_mean_norm:.6f}, center_raw_mean_norm={new_mean_norm_raw:.6f})"
                    )
                
                if not self.use_edge_age_persistence:
                    soinn_edges = {
                        int(i): {int(j): 0 for j in neighbors}
                        for i, neighbors in soinn_edges.items()
                    }
                self.class_edges[cls] = soinn_edges
                self.class_edge_reliability[cls] = self._compute_edge_reliability(clusters, soinn_edges)
                self.buffers[cls] = []
                
                if self.use_soinn_refinement:
                    logging.info(
                        f"[HC-SOINN] class {cls}: hierarchical_clusters={hierarchical_count} -> "
                        f"soinn_refined={final_count} (reduction: {hierarchical_count - final_count})"
                    )
                else:
                    logging.info(
                        f"[HC-SOINN] class {cls}: hierarchical_clusters={hierarchical_count} -> "
                        f"merged={final_count} (reduction: {hierarchical_count - final_count})"
                    )
        else:
            logging.info("[HC-SOINN] No buffer to compress.")

        # Prototypes replaced => invalidate inference cache
        self.invalidate_cache()

        # E-7: Dictionary coding & lifecycle integration at task boundary
        # Updated call chain (Agent F):
        #   dict_init()/dict_grow_for_new_classes()
        #   -> _encode_nodes_with_dict()
        #   -> dict_grow()
        #   -> _materialize_nodes()
        #   -> _update_lifecycle_states()
        #   -> compute_diagnostics()
        if getattr(self, 'use_dict_coding', False):
            try:
                # Step 1: Initialize dictionary if not yet done
                if not self._dict_initialized:
                    if hasattr(self, 'dict_init'):
                        self.dict_init()
                        self._dict_initialized = True
                        logging.info("[LifeTopoDict] Dictionary initialized.")
                    else:
                        logging.warning("[LifeTopoDict] dict_init() not available (Agent D not yet integrated).")
                else:
                    # F-2: Register new classes that lack dictionary representation
                    self.dict_grow_for_new_classes()

                # Step 2: Sparse-encode all nodes with the shared dictionary
                if hasattr(self, '_encode_nodes_with_dict'):
                    self._encode_nodes_with_dict()
                    logging.info("[LifeTopoDict] Nodes encoded with shared dictionary.")
                else:
                    logging.warning("[LifeTopoDict] _encode_nodes_with_dict() not available (Agent D not yet integrated).")

                # Step 3 (F-1): Residual-triggered dictionary growth
                growth_events = self.dict_grow()
                if growth_events:
                    # Store GTE records for later diagnostics
                    self._gte_records.extend(growth_events)
                    logging.info(
                        f"[LifeTopoDict] dict_grow() triggered {len(growth_events)} growth events."
                    )

                # Step 4: Materialize (reconstruct) node centers from D and coefficients
                if hasattr(self, '_materialize_nodes'):
                    self._materialize_nodes()
                    logging.info("[LifeTopoDict] Node centers materialized from dictionary.")

                # Step 5: Update lifecycle states (protected / plastic / inactive)
                self._update_lifecycle_states()

                # Step 6 (F-6): Compute and log diagnostics
                diagnostics = self.compute_diagnostics()

                # Step 7: Refresh inference cache with updated prototypes
                self.invalidate_cache()

            except Exception as e:
                logging.error(f"[LifeTopoDict] Lifecycle integration error in compress(): {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def predict_topk(
        self, query_features, topk: int, total_classes: int, device=None,
        use_ease_reweighting: bool = False,
        cur_task: int = 0,
        task_increment: int = 10,
        init_cls: int = 10,
        ease_alpha: float = 0.5,
        targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Handle predict topk."""
        if query_features.shape[0] == 0:
            return np.zeros((0, topk), dtype=np.int64)
            
        if device is None:
            device = torch.device("cpu")

        total_t0 = None
        if self.enable_inference_profiling:
            total_t0 = self._profile_tic(device)

        query_features = np.asarray(query_features, dtype=np.float32)
        query_dim = query_features.shape[1]
        
        classes = sorted(self.class_mu.keys())
        if len(classes) == 0:
            fallback = np.arange(min(topk, max(total_classes, 1)), dtype=np.int64)
            return np.tile(fallback, (query_features.shape[0], 1))
        
        valid_classes = [cls for cls in classes if self.class_mu[cls].shape[0] == query_dim]
        
        if not valid_classes:
            return np.zeros((query_features.shape[0], topk), dtype=np.int64)

        stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
        query_t = torch.from_numpy(query_features).to(device=device, dtype=torch.float32)
        if stage_t0 is not None:
            self._profile_toc("query_to_device", stage_t0, device)

        # Build/reuse cached prototype tensors on the target device
        stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
        self._ensure_predict_cache(device=device, query_dim=query_dim, valid_classes=valid_classes)
        if stage_t0 is not None:
            self._profile_toc("build_or_reuse_cache", stage_t0, device)

        # ---- Optimization: normalize query ONCE for the entire call ----
        stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
        q_norm = torch.nn.functional.normalize(query_t, p=2, dim=1)  # [N, D]
        if stage_t0 is not None:
            self._profile_toc("normalize_query", stage_t0, device)
        
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        def compute_distance(protos_t, labels_t=None):
            """Handle compute distance."""
            if use_ease_reweighting and cur_task > 0:
                chunk_dim = 768
                num_chunks = query_dim // chunk_dim
                
                # Reshape: [N, T, C]
                q_chunks = query_t.view(query_t.shape[0], num_chunks, chunk_dim)
                p_chunks = protos_t.view(protos_t.shape[0], num_chunks, chunk_dim)
                
                # Normalize each chunk independently
                q_chunks = torch.nn.functional.normalize(q_chunks, p=2, dim=2)
                p_chunks = torch.nn.functional.normalize(p_chunks, p=2, dim=2)
                
                sim_per_chunk = (q_chunks.unsqueeze(1) * p_chunks.unsqueeze(0)).sum(dim=-1)
                
                if labels_t is None:
                    proto_task_ids = []
                    for c in valid_classes:
                        if c < init_cls:
                            tid = 0
                        else:
                            tid = (c - init_cls) // task_increment + 1
                        proto_task_ids.append(tid)
                    proto_task_ids = torch.tensor(proto_task_ids, device=device)
                else:
                    proto_task_ids = torch.zeros_like(labels_t)
                    mask_init = labels_t < init_cls
                    proto_task_ids[mask_init] = 0
                    proto_task_ids[~mask_init] = (labels_t[~mask_init] - init_cls) // task_increment + 1
                
                M = protos_t.shape[0]
                T = num_chunks
                weights = torch.ones(M, T, device=device)
                
                task_ids_expanded = proto_task_ids.unsqueeze(1).expand(M, T)
                chunk_indices = torch.arange(T, device=device).unsqueeze(0).expand(M, T)
                
                penalty_mask = (chunk_indices != task_ids_expanded)
                factor = ease_alpha / float(cur_task)
                weights[penalty_mask] = factor
                
                weighted_sim = (sim_per_chunk * weights.unsqueeze(0)).sum(dim=2)
                
                return 1.0 - weighted_sim
                
            else:
                sim = torch.mm(q_norm, protos_t.t())
                return 1.0 - sim

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        ncm_centers_t = self._predict_cache["ncm_centers_t"]
        
        stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
        dist_ncm = compute_distance(ncm_centers_t)  # [N, C]
        if stage_t0 is not None:
            self._profile_toc("compute_ncm_distance", stage_t0, device)
        
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        all_protos_t = self._predict_cache["all_protos_t"]
        proto_labels_t = self._predict_cache["proto_labels_t"]
        proto_class_index_t = self._predict_cache["proto_class_index_t"]

        C = len(valid_classes)
        N = query_t.shape[0]

        coarse_k = self.coarse_topk
        use_coarse = (
            coarse_k is not None
            and coarse_k < C
            and not (use_ease_reweighting and cur_task > 0)
            and not getattr(self, 'use_edge_aware_scoring', False)
        )

        if use_coarse:
            # ---- Two-stage: NCM coarse filter → sub-cluster refine ----
            stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
            _, coarse_indices = torch.topk(dist_ncm, k=coarse_k, dim=1, largest=False)  # [N, coarse_k]
            candidate_set = torch.unique(coarse_indices.reshape(-1))  # union across batch

            candidate_mask_C = torch.zeros(C, dtype=torch.bool, device=device)
            candidate_mask_C[candidate_set] = True
            proto_in_candidate = candidate_mask_C[proto_class_index_t]  # [M] bool mask

            filtered_indices = proto_in_candidate.nonzero(as_tuple=True)[0]
            filtered_protos = all_protos_t[filtered_indices]               # [M', D]
            filtered_class_index = proto_class_index_t[filtered_indices]   # [M']

            dist_filtered = 1.0 - torch.mm(q_norm, filtered_protos.t())    # [N, M']

            dist_sub = torch.full((N, C), float("inf"), device=device, dtype=dist_filtered.dtype)

            if hasattr(dist_sub, "scatter_reduce_"):
                idx = filtered_class_index.view(1, -1).expand(N, -1)
                dist_sub.scatter_reduce_(1, idx, dist_filtered, reduce="amin", include_self=True)
            else:
                for ci in candidate_set.tolist():
                    mask = (filtered_class_index == ci)
                    if mask.any():
                        min_d, _ = dist_filtered[:, mask].min(dim=1)
                        dist_sub[:, ci] = min_d
            if stage_t0 is not None:
                self._profile_toc("compute_subcluster_distance", stage_t0, device)

        else:
            # ---- Original: evaluate all prototypes ----
            stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
            dist_proto_all = compute_distance(all_protos_t, proto_labels_t)  # [N, M]

            dist_sub = torch.full((N, C), float("inf"), device=device, dtype=dist_proto_all.dtype)

            if hasattr(dist_sub, "scatter_reduce_"):
                idx = proto_class_index_t.view(1, -1).expand(N, -1)  # [N, M]
                dist_sub.scatter_reduce_(1, idx, dist_proto_all, reduce="amin", include_self=True)
            else:
                for i, cls in enumerate(valid_classes):
                    mask = (proto_labels_t == cls)
                    if mask.any():
                        min_d, _ = dist_proto_all[:, mask].min(dim=1)
                        dist_sub[:, i] = min_d
                    else:
                        dist_sub[:, i] = dist_ncm[:, i]
            if stage_t0 is not None:
                self._profile_toc("compute_subcluster_distance", stage_t0, device)
        
        # score = alpha * d_ncm + (1 - alpha) * d_sub
        stage_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
        final_scores = self.alpha * dist_ncm + (1.0 - self.alpha) * dist_sub
        if getattr(self, 'use_edge_aware_scoring', False) and 'dist_proto_all' in locals():
            edge_adjustment = self._compute_edge_score_adjustment(dist_proto_all, valid_classes, device)
            final_scores = final_scores + edge_adjustment

        trace_enabled = bool(
            getattr(self, 'enable_prediction_trace', False)
            or getattr(self, 'use_raw_fallback_gate', False)
        )
        valid_classes_np = np.array(valid_classes, dtype=np.int64)
        trace_info = None

        if trace_enabled:
            if 'dist_proto_all' not in locals():
                stage_trace_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
                dist_proto_all = compute_distance(all_protos_t, proto_labels_t)
                if stage_trace_t0 is not None:
                    self._profile_toc("compute_trace_proto_distance", stage_trace_t0, device)

            rank_k = min(2, C)
            compact_vals, compact_indices = torch.topk(final_scores, k=rank_k, dim=1, largest=False)
            compact_top1_idx_t = compact_indices[:, 0]
            compact_top1_score_t = compact_vals[:, 0]
            if rank_k > 1:
                compact_top2_idx_t = compact_indices[:, 1]
                compact_top2_score_t = compact_vals[:, 1]
                compact_margin_t = compact_top2_score_t - compact_top1_score_t
            else:
                compact_top2_idx_t = compact_top1_idx_t
                compact_top2_score_t = compact_top1_score_t
                compact_margin_t = torch.full_like(compact_top1_score_t, float("inf"))

            compact_top1_idx_np = compact_top1_idx_t.detach().cpu().numpy()
            compact_top2_idx_np = compact_top2_idx_t.detach().cpu().numpy()
            compact_top1_score_np = compact_top1_score_t.detach().cpu().numpy()
            compact_top2_score_np = compact_top2_score_t.detach().cpu().numpy()
            compact_scores_np = final_scores.detach().cpu().numpy()
            compact_margin_np = compact_margin_t.detach().cpu().numpy()
            compact_top1_labels_np = valid_classes_np[compact_top1_idx_np]
            compact_top2_labels_np = valid_classes_np[compact_top2_idx_np]

            proto_class_index_np = proto_class_index_t.detach().cpu().numpy()
            proto_node_index_np = self._predict_cache["proto_node_index_t"].detach().cpu().numpy()
            dist_proto_np = dist_proto_all.detach().cpu().numpy()
            proto_positions_by_class = {
                ci: np.where(proto_class_index_np == ci)[0]
                for ci in range(C)
            }
            selected_node_np = np.full(N, -1, dtype=np.int64)
            selected_count_np = np.zeros(N, dtype=np.float32)
            selected_residual_np = np.zeros(N, dtype=np.float32)
            selected_distance_np = np.full(N, np.nan, dtype=np.float32)
            selected_state_np = np.array(["unknown"] * N, dtype=object)
            selected_atoms: List[List[int]] = [[] for _ in range(N)]
            selected_atom_inactive_ratio_np = np.zeros(N, dtype=np.float32)
            selected_atom_plastic_ratio_np = np.zeros(N, dtype=np.float32)
            selected_atom_protected_ratio_np = np.zeros(N, dtype=np.float32)
            selected_atom_weighted_inactive_ratio_np = np.zeros(N, dtype=np.float32)
            selected_atom_weighted_protected_ratio_np = np.zeros(N, dtype=np.float32)
            selected_atom_old_support_mean_np = np.zeros(N, dtype=np.float32)
            selected_atom_weighted_old_support_np = np.zeros(N, dtype=np.float32)
            selected_atom_usage_ema_mean_np = np.zeros(N, dtype=np.float32)
            selected_atom_weighted_usage_ema_np = np.zeros(N, dtype=np.float32)
            selected_atom_abs_sum_np = np.zeros(N, dtype=np.float32)
            selected_atom_entropy_np = np.zeros(N, dtype=np.float32)
            for row, class_index in enumerate(compact_top1_idx_np):
                positions = proto_positions_by_class.get(int(class_index))
                if positions is None or positions.size == 0:
                    continue
                local_offset = int(np.argmin(dist_proto_np[row, positions]))
                proto_pos = int(positions[local_offset])
                node_idx = int(proto_node_index_np[proto_pos])
                cls = int(valid_classes[int(class_index)])
                selected_node_np[row] = node_idx
                selected_distance_np[row] = float(dist_proto_np[row, proto_pos])
                if node_idx >= 0 and node_idx < len(self.class_clusters.get(cls, [])):
                    node = self.class_clusters[cls][node_idx]
                    selected_count_np[row] = float(getattr(node, "count", 0.0))
                    selected_residual_np[row] = float(getattr(node, "residual", 0.0))
                    selected_state_np[row] = str(getattr(node, "node_state", "plastic"))
                    coeff = getattr(node, "coeff", None)
                    if isinstance(coeff, np.ndarray) and coeff.size > 0:
                        coeff_flat = coeff.reshape(-1)
                        abs_coeff = np.abs(coeff_flat)
                        nonzero = np.flatnonzero(abs_coeff > 1e-6)
                        if nonzero.size > 0:
                            ordered = nonzero[np.argsort(-abs_coeff[nonzero])]
                            top_atoms = ordered[: self.dict_sparse_k]
                            selected_atoms[row] = [int(i) for i in top_atoms]
                            top_weights = abs_coeff[top_atoms].astype(np.float64, copy=False)
                            weight_sum = float(np.sum(top_weights))
                            if weight_sum > 0.0:
                                probs = top_weights / weight_sum
                                states = [str(self.atom_states[int(i)]) if int(i) < len(self.atom_states) else "plastic" for i in top_atoms]
                                old_support_vals = np.asarray(
                                    [float(self.old_support.get(int(i), 0.0)) for i in top_atoms],
                                    dtype=np.float64,
                                )
                                usage_vals = np.asarray(
                                    [float(self.atom_usage_ema.get(int(i), 0.0)) for i in top_atoms],
                                    dtype=np.float64,
                                )
                                inactive_mask = np.asarray([s == "inactive" for s in states], dtype=bool)
                                protected_mask = np.asarray([s == "protected" for s in states], dtype=bool)
                                plastic_mask = np.asarray([s == "plastic" for s in states], dtype=bool)
                                denom = float(len(top_atoms))
                                selected_atom_inactive_ratio_np[row] = float(np.mean(inactive_mask)) if denom > 0 else 0.0
                                selected_atom_plastic_ratio_np[row] = float(np.mean(plastic_mask)) if denom > 0 else 0.0
                                selected_atom_protected_ratio_np[row] = float(np.mean(protected_mask)) if denom > 0 else 0.0
                                selected_atom_weighted_inactive_ratio_np[row] = float(np.sum(probs[inactive_mask]))
                                selected_atom_weighted_protected_ratio_np[row] = float(np.sum(probs[protected_mask]))
                                selected_atom_old_support_mean_np[row] = float(np.mean(old_support_vals)) if old_support_vals.size > 0 else 0.0
                                selected_atom_weighted_old_support_np[row] = float(np.sum(probs * old_support_vals))
                                selected_atom_usage_ema_mean_np[row] = float(np.mean(usage_vals)) if usage_vals.size > 0 else 0.0
                                selected_atom_weighted_usage_ema_np[row] = float(np.sum(probs * usage_vals))
                                selected_atom_abs_sum_np[row] = float(weight_sum)
                                selected_atom_entropy_np[row] = float(-np.sum(probs * np.log(probs + 1e-12)))

            fallback_scores = None
            fallback_available_np = np.zeros(N, dtype=bool)
            fallback_top1_labels_np = np.full(N, -1, dtype=np.int64)
            fallback_top2_labels_np = np.full(N, -1, dtype=np.int64)
            fallback_top1_distance_np = np.full(N, np.nan, dtype=np.float32)
            fallback_top2_distance_np = np.full(N, np.nan, dtype=np.float32)
            fallback_margin_np = np.full(N, np.nan, dtype=np.float32)
            raw_top1_score_np = np.full(N, -np.inf, dtype=np.float32)
            raw_top2_score_np = np.full(N, -np.inf, dtype=np.float32)
            compact_rank_of_raw_top1_np = np.full(N, 999, dtype=np.float32)
            raw_rank_of_compact_top1_np = np.full(N, 999, dtype=np.float32)
            compact_score_on_raw_top1_np = np.zeros(N, dtype=np.float32)
            raw_score_on_compact_top1_np = np.zeros(N, dtype=np.float32)
            score_cross_gap_np = np.zeros(N, dtype=np.float32)
            selected_class_age_np = np.zeros(N, dtype=np.float32)
            raw_class_age_np = np.zeros(N, dtype=np.float32)
            task_relation_code_np = np.zeros(N, dtype=np.float32)
            fallback_top1_node_np = np.full(N, -1, dtype=np.int64)
            fallback_top1_residual_np = np.zeros(N, dtype=np.float32)
            fallback_top1_count_np = np.zeros(N, dtype=np.float32)
            fallback_top1_state_np = np.array(["unknown"] * N, dtype=object)
            fallback_gate_t = torch.zeros(N, dtype=torch.bool, device=device)

            fallback_requested = bool(
                getattr(self, 'use_raw_fallback_gate', False)
                and getattr(self, 'raw_fallback_per_class', 0) > 0
            )
            if fallback_requested:
                self._ensure_raw_fallback_predict_cache(
                    device=device,
                    query_dim=query_dim,
                    valid_classes=valid_classes,
                )
                raw_cache = self._raw_fallback_predict_cache
                fallback_centers_t = raw_cache["fallback_centers_t"]
                fallback_labels_t = raw_cache["fallback_labels_t"]
                fallback_class_index_t = raw_cache["fallback_class_index_t"]
                if fallback_centers_t.shape[0] > 0:
                    stage_fallback_t0 = self._profile_tic(device) if self.enable_inference_profiling else None
                    dist_fallback_all = compute_distance(fallback_centers_t, fallback_labels_t)
                    fallback_scores = torch.full(
                        (N, C),
                        float("inf"),
                        device=device,
                        dtype=dist_fallback_all.dtype,
                    )
                    if hasattr(fallback_scores, "scatter_reduce_"):
                        idx = fallback_class_index_t.view(1, -1).expand(N, -1)
                        fallback_scores.scatter_reduce_(
                            1, idx, dist_fallback_all, reduce="amin", include_self=True
                        )
                    else:
                        for ci in range(C):
                            mask = (fallback_class_index_t == ci)
                            if mask.any():
                                min_d, _ = dist_fallback_all[:, mask].min(dim=1)
                                fallback_scores[:, ci] = min_d

                    fallback_available_t = torch.isfinite(fallback_scores).any(dim=1)
                    fallback_available_np = fallback_available_t.detach().cpu().numpy()
                    fallback_rank_k = min(2, C)
                    fallback_vals_t, fallback_idx_t = torch.topk(
                        fallback_scores, k=fallback_rank_k, dim=1, largest=False
                    )
                    fallback_best_scores_t = fallback_vals_t[:, 0]
                    fallback_best_idx_t = fallback_idx_t[:, 0]
                    fallback_best_idx_np = fallback_best_idx_t.detach().cpu().numpy()
                    fallback_top1_labels_np[fallback_available_np] = valid_classes_np[
                        fallback_best_idx_np[fallback_available_np]
                    ]
                    fallback_top1_distance_np[fallback_available_np] = (
                        fallback_best_scores_t.detach().cpu().numpy()[fallback_available_np]
                    )
                    if fallback_rank_k > 1:
                        fallback_second_idx_np = fallback_idx_t[:, 1].detach().cpu().numpy()
                        fallback_second_dist_np = fallback_vals_t[:, 1].detach().cpu().numpy()
                        finite_second = np.logical_and(fallback_available_np, np.isfinite(fallback_second_dist_np))
                        fallback_top2_labels_np[finite_second] = valid_classes_np[
                            fallback_second_idx_np[finite_second]
                        ]
                        fallback_top2_distance_np[finite_second] = fallback_second_dist_np[finite_second]
                        fallback_margin_np[finite_second] = (
                            fallback_top2_distance_np[finite_second]
                            - fallback_top1_distance_np[finite_second]
                        )

                    fallback_dist_np = dist_fallback_all.detach().cpu().numpy()
                    fallback_scores_np = fallback_scores.detach().cpu().numpy()
                    fallback_class_index_np = fallback_class_index_t.detach().cpu().numpy()
                    fallback_node_ids_np = raw_cache["fallback_node_ids_np"]
                    fallback_counts_np = raw_cache["fallback_counts_np"]
                    fallback_residuals_np = raw_cache["fallback_residuals_np"]
                    fallback_state_codes_np = raw_cache["fallback_state_codes_np"]
                    fallback_state_lookup = {0: "inactive", 1: "plastic", 2: "protected"}
                    for row in np.where(fallback_available_np)[0]:
                        best_ci = int(fallback_best_idx_np[row])
                        positions = np.where(fallback_class_index_np == best_ci)[0]
                        if positions.size == 0:
                            continue
                        local_offset = int(np.argmin(fallback_dist_np[row, positions]))
                        best_pos = int(positions[local_offset])
                        fallback_top1_node_np[row] = int(fallback_node_ids_np[best_pos])
                        fallback_top1_residual_np[row] = float(fallback_residuals_np[best_pos])
                        fallback_top1_count_np[row] = float(fallback_counts_np[best_pos])
                        fallback_top1_state_np[row] = str(fallback_state_lookup.get(int(fallback_state_codes_np[best_pos]), "plastic"))

                    boundary = getattr(self, "known_classes_before_task", None)
                    if boundary is None:
                        boundary = 0
                    boundary = int(boundary)
                    for row in np.where(fallback_available_np)[0]:
                        raw_idx = int(fallback_best_idx_np[row])
                        compact_idx = int(compact_top1_idx_np[row])
                        if 0 <= raw_idx < C:
                            raw_dist = float(fallback_scores_np[row, raw_idx])
                            compact_dist_on_raw = float(compact_scores_np[row, raw_idx])
                            raw_top1_score_np[row] = -raw_dist if np.isfinite(raw_dist) else -np.inf
                            compact_score_on_raw_top1_np[row] = (
                                -compact_dist_on_raw if np.isfinite(compact_dist_on_raw) else 0.0
                            )
                            compact_rank_of_raw_top1_np[row] = float(
                                1 + np.sum(compact_scores_np[row] < compact_dist_on_raw)
                            )
                        if 0 <= compact_idx < C:
                            compact_dist = float(compact_scores_np[row, compact_idx])
                            raw_dist_on_compact = float(fallback_scores_np[row, compact_idx])
                            raw_score_on_compact_top1_np[row] = (
                                -raw_dist_on_compact if np.isfinite(raw_dist_on_compact) else 0.0
                            )
                            raw_rank_of_compact_top1_np[row] = float(
                                1 + np.sum(fallback_scores_np[row] < raw_dist_on_compact)
                            )
                        if np.isfinite(fallback_top2_distance_np[row]):
                            raw_top2_score_np[row] = -float(fallback_top2_distance_np[row])
                        compact_gap = (-float(compact_top1_score_np[row])) - compact_score_on_raw_top1_np[row]
                        raw_gap = raw_top1_score_np[row] - raw_score_on_compact_top1_np[row]
                        if np.isfinite(raw_gap) and np.isfinite(compact_gap):
                            score_cross_gap_np[row] = float(raw_gap - compact_gap)
                        selected_cls = int(compact_top1_labels_np[row])
                        raw_cls = int(fallback_top1_labels_np[row])
                        selected_old = boundary > 0 and selected_cls < boundary
                        raw_old = boundary > 0 and raw_cls < boundary
                        selected_class_age_np[row] = 1.0 if selected_old else 0.0
                        raw_class_age_np[row] = 1.0 if raw_old else 0.0
                        if selected_old and (not raw_old):
                            task_relation_code_np[row] = 1.0  # old_to_new
                        elif (not selected_old) and raw_old:
                            task_relation_code_np[row] = 2.0  # new_to_old
                        else:
                            task_relation_code_np[row] = 0.0  # same side of current boundary

                    gate_type = self._canonical_fallback_gate_type(
                        str(getattr(self, "fallback_gate_type", "rule")).lower()
                    )
                    if gate_type == "random":
                        random_mask_np = self._raw_fallback_rng.rand(N) < self.fallback_gate_random_rate
                        fallback_gate_t = torch.from_numpy(
                            np.logical_and(random_mask_np, fallback_available_np)
                        ).to(device=device, dtype=torch.bool)
                    elif gate_type == "oracle":
                        if targets is not None:
                            targets_np = np.asarray(targets).reshape(-1)
                            compact_wrong_np = compact_top1_labels_np != targets_np
                            fallback_correct_np = fallback_top1_labels_np == targets_np
                            gate_np = np.logical_and.reduce((
                                fallback_available_np,
                                compact_wrong_np,
                                fallback_correct_np,
                            ))
                            fallback_gate_t = torch.from_numpy(gate_np).to(device=device, dtype=torch.bool)
                        else:
                            fallback_gate_t = torch.zeros(N, dtype=torch.bool, device=device)
                    elif gate_type in ("static", "always"):
                        fallback_gate_t = torch.from_numpy(fallback_available_np).to(
                            device=device, dtype=torch.bool
                        )
                    elif gate_type in self._learned_fallback_gate_types():
                        gate_model = getattr(self, "_raw_fallback_gate_model", {}) or {}
                        gate_model_type = self._canonical_fallback_gate_type(
                            str(gate_model.get("type", "")).lower()
                        )
                        gate_dirty = bool(getattr(self, "_raw_fallback_gate_model_dirty", True))
                        coeff = gate_model.get("coeff")
                        weights = gate_model.get("weights")
                        bias = float(gate_model.get("bias", 0.0))
                        if coeff is None and isinstance(weights, np.ndarray):
                            coeff = np.concatenate(
                                [np.asarray([bias], dtype=np.float32), weights.astype(np.float32, copy=False)],
                                axis=0,
                            )
                        feature_mean = gate_model.get("feature_mean")
                        feature_std = gate_model.get("feature_std")
                        threshold = float(gate_model.get("threshold", 0.0))
                        if (
                            not gate_dirty
                            and gate_model_type == gate_type
                            and isinstance(feature_mean, np.ndarray)
                            and isinstance(feature_std, np.ndarray)
                        ):
                            gate_features, _ = self._build_raw_fallback_gate_features(
                                compact_margin=compact_margin_np,
                                selected_residual=selected_residual_np,
                                selected_distance=selected_distance_np,
                                fallback_distance=fallback_top1_distance_np,
                                selected_count=selected_count_np,
                                fallback_residual=fallback_top1_residual_np,
                                fallback_count=fallback_top1_count_np,
                                selected_state=selected_state_np,
                                compact_top1_score=compact_top1_score_np,
                                compact_top2_score=compact_top2_score_np,
                                fallback_margin=fallback_margin_np,
                                raw_top1_score=raw_top1_score_np,
                                raw_top2_score=raw_top2_score_np,
                                selected_atom_weighted_inactive_ratio=selected_atom_weighted_inactive_ratio_np,
                                selected_atom_old_support_mean=selected_atom_old_support_mean_np,
                                selected_atom_weighted_old_support=selected_atom_weighted_old_support_np,
                                selected_atom_usage_ema_mean=selected_atom_usage_ema_mean_np,
                                selected_atom_weighted_usage_ema=selected_atom_weighted_usage_ema_np,
                                selected_atom_entropy=selected_atom_entropy_np,
                                compact_raw_disagree=(compact_top1_labels_np != fallback_top1_labels_np).astype(np.float32),
                                compact_rank_of_raw_top1=compact_rank_of_raw_top1_np,
                                raw_rank_of_compact_top1=raw_rank_of_compact_top1_np,
                                compact_score_on_raw_top1=compact_score_on_raw_top1_np,
                                raw_score_on_compact_top1=raw_score_on_compact_top1_np,
                                score_cross_gap=score_cross_gap_np,
                                selected_class_age=selected_class_age_np,
                                raw_class_age=raw_class_age_np,
                                task_relation_code=task_relation_code_np,
                                fallback_available=fallback_available_np.astype(np.float32),
                            )
                            gate_features = (gate_features - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)
                            if gate_type in ("ridge", "logistic") and isinstance(coeff, np.ndarray):
                                gate_scores = np.concatenate(
                                    [
                                        np.ones((N, 1), dtype=np.float32),
                                        gate_features.astype(np.float32, copy=False),
                                    ],
                                    axis=1,
                                ) @ coeff.astype(np.float32, copy=False)
                            elif gate_type in ("delta_ridge", "delta_logistic", "delta_ridge_pair") and isinstance(coeff, np.ndarray):
                                gate_scores = np.concatenate(
                                    [
                                        np.ones((N, 1), dtype=np.float32),
                                        gate_features.astype(np.float32, copy=False),
                                    ],
                                    axis=1,
                                ) @ coeff.astype(np.float32, copy=False)
                            elif gate_type == "utility_table":
                                gate_scores = np.zeros(N, dtype=np.float32)
                            elif gate_type == "dual_logistic":
                                compact_estimator = gate_model.get("compact_estimator")
                                raw_estimator = gate_model.get("raw_estimator")
                                compact_feature_indices = gate_model.get("compact_feature_indices")
                                raw_feature_indices = gate_model.get("raw_feature_indices")
                                if (
                                    compact_estimator is not None
                                    and raw_estimator is not None
                                    and hasattr(compact_estimator, "predict_proba")
                                    and hasattr(raw_estimator, "predict_proba")
                                    and isinstance(compact_feature_indices, np.ndarray)
                                    and isinstance(raw_feature_indices, np.ndarray)
                                ):
                                    X_gate = gate_features.astype(np.float32, copy=False)
                                    p_compact = compact_estimator.predict_proba(
                                        X_gate[:, compact_feature_indices.astype(np.int64)]
                                    )[:, 1]
                                    p_raw = raw_estimator.predict_proba(
                                        X_gate[:, raw_feature_indices.astype(np.int64)]
                                    )[:, 1]
                                    gate_scores = p_raw - p_compact
                                else:
                                    gate_scores = np.full(N, -np.inf, dtype=np.float32)
                            else:
                                estimator = gate_model.get("estimator")
                                if estimator is not None and hasattr(estimator, "predict_proba"):
                                    prob = estimator.predict_proba(
                                        gate_features.astype(np.float32, copy=False)
                                    )[:, 1]
                                    if gate_type == "benefit_harm_logistic":
                                        gate_scores = 2.0 * prob - 1.0
                                    else:
                                        gate_scores = prob
                                elif estimator is not None and hasattr(estimator, "predict"):
                                    gate_scores = estimator.predict(
                                        gate_features.astype(np.float32, copy=False)
                                    )
                                else:
                                    gate_scores = np.full(N, -np.inf, dtype=np.float32)
                            candidate_params = gate_model.get("candidate_params", {}) or {}
                            candidate_rule = str(gate_model.get("candidate_rule", "none"))
                            candidate_data = {
                                "fallback_available": fallback_available_np,
                                "compact_top1": compact_top1_labels_np,
                                "fallback_top1": fallback_top1_labels_np,
                                "compact_margin": compact_margin_np,
                                "fallback_margin": fallback_margin_np,
                                "selected_residual": selected_residual_np,
                                "compact_rank_of_raw_top1": compact_rank_of_raw_top1_np,
                                "raw_rank_of_compact_top1": raw_rank_of_compact_top1_np,
                                "score_cross_gap": score_cross_gap_np,
                            }
                            candidate_mask_np = self._build_raw_fallback_candidate_mask(
                                candidate_data,
                                params=candidate_params,
                                rule=candidate_rule,
                            )
                            pair_mode = str(gate_model.get("pair_table_mode", "none")).lower()
                            if pair_mode not in ("", "none"):
                                pair_scores, node_scores = self._lookup_fallback_utility_tables(
                                    gate_model,
                                    compact_top1=compact_top1_labels_np,
                                    fallback_top1=fallback_top1_labels_np,
                                    selected_node=selected_node_np,
                                    task_relation_code=task_relation_code_np,
                                )
                                if pair_mode in ("class_pair", "class_pair_only", "class_pair_task", "class_pair_task_only"):
                                    gate_scores = gate_scores + float(gate_model.get("fallback_pair_table_lambda", 0.5)) * pair_scores
                                elif pair_mode in ("node_pair", "node_pair_only"):
                                    gate_scores = gate_scores + float(gate_model.get("fallback_node_table_lambda", 0.5)) * node_scores
                                else:
                                    gate_scores = gate_scores + float(gate_model.get("fallback_pair_table_lambda", 0.5)) * pair_scores
                                    gate_scores = gate_scores + float(gate_model.get("fallback_node_table_lambda", 0.5)) * node_scores
                            gate_np = np.logical_and.reduce((
                                gate_scores >= threshold,
                                fallback_available_np,
                                candidate_mask_np,
                            ))
                            fallback_gate_t = torch.from_numpy(gate_np).to(device=device, dtype=torch.bool)
                        else:
                            residual_t = torch.from_numpy(selected_residual_np).to(
                                device=device, dtype=torch.float32
                            )
                            fallback_gate_t = (
                                (compact_margin_t <= self.fallback_gate_margin_threshold)
                                & (residual_t >= self.fallback_gate_residual_threshold)
                                & torch.from_numpy(fallback_available_np).to(device=device, dtype=torch.bool)
                            )
                    else:
                        residual_t = torch.from_numpy(selected_residual_np).to(
                            device=device, dtype=torch.float32
                        )
                        fallback_gate_t = (
                            (compact_margin_t <= self.fallback_gate_margin_threshold)
                            & (residual_t >= self.fallback_gate_residual_threshold)
                            & torch.from_numpy(fallback_available_np).to(device=device, dtype=torch.bool)
                        )

                    if fallback_gate_t.any():
                        gated_scores = fallback_scores[fallback_gate_t]
                        finite_rows = torch.isfinite(gated_scores).any(dim=1)
                        if finite_rows.any():
                            combined_scores = final_scores.clone()
                            gate_rows = fallback_gate_t.nonzero(as_tuple=True)[0]
                            valid_gate_rows = gate_rows[finite_rows]
                            combined_scores[valid_gate_rows] = gated_scores[finite_rows]
                            final_scores = combined_scores
                    if stage_fallback_t0 is not None:
                        self._profile_toc("raw_fallback_gate", stage_fallback_t0, device)

            fallback_gate_np = fallback_gate_t.detach().cpu().numpy()
            trace_info = {
                "compact_top1_labels": compact_top1_labels_np,
                "compact_top2_labels": compact_top2_labels_np,
                "compact_top1_score": compact_top1_score_np,
                "compact_top2_score": compact_top2_score_np,
                "compact_margin": compact_margin_np,
                "selected_node": selected_node_np,
                "selected_count": selected_count_np,
                "selected_distance": selected_distance_np,
                "selected_residual": selected_residual_np,
                "selected_state": selected_state_np,
                "selected_atoms": selected_atoms,
                "selected_atom_inactive_ratio": selected_atom_inactive_ratio_np,
                "selected_atom_plastic_ratio": selected_atom_plastic_ratio_np,
                "selected_atom_protected_ratio": selected_atom_protected_ratio_np,
                "selected_atom_weighted_inactive_ratio": selected_atom_weighted_inactive_ratio_np,
                "selected_atom_weighted_protected_ratio": selected_atom_weighted_protected_ratio_np,
                "selected_atom_old_support_mean": selected_atom_old_support_mean_np,
                "selected_atom_weighted_old_support": selected_atom_weighted_old_support_np,
                "selected_atom_usage_ema_mean": selected_atom_usage_ema_mean_np,
                "selected_atom_weighted_usage_ema": selected_atom_weighted_usage_ema_np,
                "selected_atom_abs_sum": selected_atom_abs_sum_np,
                "selected_atom_entropy": selected_atom_entropy_np,
                "fallback_available": fallback_available_np,
                "fallback_top1_labels": fallback_top1_labels_np,
                "fallback_top2_labels": fallback_top2_labels_np,
                "fallback_top1_distance": fallback_top1_distance_np,
                "fallback_top2_distance": fallback_top2_distance_np,
                "fallback_margin": fallback_margin_np,
                "raw_top1_score": raw_top1_score_np,
                "raw_top2_score": raw_top2_score_np,
                "compact_rank_of_raw_top1": compact_rank_of_raw_top1_np,
                "raw_rank_of_compact_top1": raw_rank_of_compact_top1_np,
                "compact_score_on_raw_top1": compact_score_on_raw_top1_np,
                "raw_score_on_compact_top1": raw_score_on_compact_top1_np,
                "score_cross_gap": score_cross_gap_np,
                "selected_class_age": selected_class_age_np,
                "raw_class_age": raw_class_age_np,
                "task_relation_code": task_relation_code_np,
                "fallback_top1_node": fallback_top1_node_np,
                "fallback_top1_count": fallback_top1_count_np,
                "fallback_top1_residual": fallback_top1_residual_np,
                "fallback_top1_state": fallback_top1_state_np,
                "fallback_gate": fallback_gate_np,
            }

        k = min(topk, len(valid_classes))
        _, indices = torch.topk(final_scores, k=k, dim=1, largest=False)
        indices = indices.cpu().numpy()

        top_preds = valid_classes_np[indices]  # [N, topk]

        if trace_info is not None:
            targets_np = None if targets is None else np.asarray(targets).reshape(-1)
            samples = int(top_preds.shape[0])
            stats = self._raw_fallback_eval_stats
            compact_pred = trace_info["compact_top1_labels"]
            fallback_pred = trace_info["fallback_top1_labels"]
            final_pred = top_preds[:, 0]
            fallback_gate = trace_info["fallback_gate"].astype(bool)
            fallback_available = trace_info["fallback_available"].astype(bool)
            prediction_changed = final_pred != compact_pred
            compact_raw_disagree = np.logical_and(
                fallback_available,
                fallback_pred != compact_pred,
            )

            stats["samples"] += float(samples)
            stats["fallback_used"] += float(np.sum(fallback_gate))
            stats["prediction_changed"] += float(np.sum(prediction_changed))
            stats["compact_raw_disagree"] += float(np.sum(compact_raw_disagree))
            stats["fallback_available"] += float(np.sum(fallback_available))
            stats["margin_sum"] += float(np.nansum(trace_info["compact_margin"]))
            stats["residual_sum"] += float(np.nansum(trace_info["selected_residual"]))
            if np.any(fallback_gate):
                stats["fallback_margin_sum"] += float(np.nansum(trace_info["compact_margin"][fallback_gate]))
                stats["fallback_residual_sum"] += float(np.nansum(trace_info["selected_residual"][fallback_gate]))

            if targets_np is not None and targets_np.shape[0] == samples:
                compact_correct = compact_pred == targets_np
                final_correct = final_pred == targets_np
                fallback_correct = fallback_pred == targets_np
                benefit_mask = np.logical_and.reduce((fallback_available, ~compact_correct, fallback_correct))
                harm_mask = np.logical_and.reduce((fallback_available, compact_correct, ~fallback_correct))
                neutral_mask = np.logical_and(fallback_available, ~(benefit_mask | harm_mask))
                stats["compact_correct"] += float(np.sum(compact_correct))
                stats["final_correct"] += float(np.sum(final_correct))
                stats["fallback_correct"] += float(np.sum(np.logical_and(fallback_available, fallback_correct)))
                stats["compact_correct_when_used"] += float(np.sum(np.logical_and(fallback_gate, compact_correct)))
                stats["fallback_correct_when_used"] += float(np.sum(np.logical_and(fallback_gate, fallback_correct)))
                stats["oracle_improvable"] += float(np.sum(np.logical_and(~compact_correct, fallback_correct)))
                stats["benefit_total"] += float(np.sum(benefit_mask))
                stats["harm_total"] += float(np.sum(harm_mask))
                stats["neutral_total"] += float(np.sum(neutral_mask))
                stats["benefit_selected"] += float(np.sum(np.logical_and(fallback_gate, benefit_mask)))
                stats["harm_selected"] += float(np.sum(np.logical_and(fallback_gate, harm_mask)))
                stats["neutral_selected"] += float(np.sum(np.logical_and(fallback_gate, neutral_mask)))

            if getattr(self, "enable_prediction_trace", False):
                for row in range(samples):
                    record = {
                        "compact_top1": int(compact_pred[row]),
                        "compact_top2": int(trace_info["compact_top2_labels"][row]),
                        "compact_top1_score": float(trace_info["compact_top1_score"][row]),
                        "compact_top2_score": float(trace_info["compact_top2_score"][row]),
                        "compact_margin": float(trace_info["compact_margin"][row]),
                        "nearest_compact_node": int(trace_info["selected_node"][row]),
                        "nearest_compact_count": float(trace_info["selected_count"][row]),
                        "selected_node_count": float(trace_info["selected_count"][row]),
                        "nearest_compact_distance": float(trace_info["selected_distance"][row]),
                        "selected_node_residual": float(trace_info["selected_residual"][row]),
                        "selected_node_state": str(trace_info["selected_state"][row]),
                        "selected_atom_topk": trace_info["selected_atoms"][row],
                        "selected_atom_inactive_ratio": float(trace_info["selected_atom_inactive_ratio"][row]),
                        "selected_atom_plastic_ratio": float(trace_info["selected_atom_plastic_ratio"][row]),
                        "selected_atom_protected_ratio": float(trace_info["selected_atom_protected_ratio"][row]),
                        "selected_atom_weighted_inactive_ratio": float(trace_info["selected_atom_weighted_inactive_ratio"][row]),
                        "selected_atom_weighted_protected_ratio": float(trace_info["selected_atom_weighted_protected_ratio"][row]),
                        "selected_atom_old_support_mean": float(trace_info["selected_atom_old_support_mean"][row]),
                        "selected_atom_weighted_old_support": float(trace_info["selected_atom_weighted_old_support"][row]),
                        "selected_atom_usage_ema_mean": float(trace_info["selected_atom_usage_ema_mean"][row]),
                        "selected_atom_weighted_usage_ema": float(trace_info["selected_atom_weighted_usage_ema"][row]),
                        "selected_atom_abs_sum": float(trace_info["selected_atom_abs_sum"][row]),
                        "selected_atom_entropy": float(trace_info["selected_atom_entropy"][row]),
                        "fallback_available": bool(fallback_available[row]),
                        "fallback_used": bool(fallback_gate[row]),
                        "fallback_top1": int(fallback_pred[row]),
                        "fallback_top2": int(trace_info["fallback_top2_labels"][row]),
                        "raw_top1_class": int(fallback_pred[row]),
                        "selected_class": int(compact_pred[row]),
                        "fallback_node": int(trace_info["fallback_top1_node"][row]),
                        "fallback_count": float(trace_info["fallback_top1_count"][row]),
                        "fallback_distance": float(trace_info["fallback_top1_distance"][row]),
                        "fallback_top2_distance": float(trace_info["fallback_top2_distance"][row]),
                        "fallback_margin": float(trace_info["fallback_margin"][row]),
                        "raw_top1_score": float(trace_info["raw_top1_score"][row]),
                        "raw_top2_score": float(trace_info["raw_top2_score"][row]),
                        "raw_margin": float(trace_info["fallback_margin"][row]),
                        "raw_distance_to_top1": float(trace_info["fallback_top1_distance"][row]),
                        "raw_distance_to_top2": float(trace_info["fallback_top2_distance"][row]),
                        "compact_rank_of_raw_top1": float(trace_info["compact_rank_of_raw_top1"][row]),
                        "raw_rank_of_compact_top1": float(trace_info["raw_rank_of_compact_top1"][row]),
                        "compact_score_on_raw_top1": float(trace_info["compact_score_on_raw_top1"][row]),
                        "raw_score_on_compact_top1": float(trace_info["raw_score_on_compact_top1"][row]),
                        "score_cross_gap": float(trace_info["score_cross_gap"][row]),
                        "selected_class_age": float(trace_info["selected_class_age"][row]),
                        "raw_class_age": float(trace_info["raw_class_age"][row]),
                        "class_age": float(trace_info["selected_class_age"][row]),
                        "task_relation_code": float(trace_info["task_relation_code"][row]),
                        "fallback_residual": float(trace_info["fallback_top1_residual"][row]),
                        "fallback_state": str(trace_info["fallback_top1_state"][row]),
                        "final_top1": int(final_pred[row]),
                    }
                    relation_code = int(trace_info["task_relation_code"][row])
                    if relation_code == 1:
                        record["task_relation"] = "old_to_new"
                    elif relation_code == 2:
                        record["task_relation"] = "new_to_old"
                    else:
                        record["task_relation"] = "same_task"
                    if targets_np is not None and targets_np.shape[0] == samples:
                        record["target"] = int(targets_np[row])
                        record["compact_correct"] = bool(compact_pred[row] == targets_np[row])
                        record["fallback_correct"] = bool(fallback_pred[row] == targets_np[row])
                        record["final_correct"] = bool(final_pred[row] == targets_np[row])
                    self._prediction_trace_records.append(record)

            self._last_fallback_stats = self._summarize_raw_fallback_eval_stats()

        if stage_t0 is not None:
            self._profile_toc("fuse_and_topk", stage_t0, device)

        if total_t0 is not None:
            total_dt = self._profile_toc("predict_topk_total", total_t0, device)
            self._profile_stats["calls"] += 1
            self._profile_stats["samples"] += int(query_features.shape[0])
            self._profile_stats["total_sec"] += total_dt
        
        return top_preds

    def predict_class_logits(
        self,
        query_features,
        total_classes: int,
        device=None,
    ) -> torch.Tensor:
        """
        Return per-class logits aligned to [0, total_classes).
        Logits are the negative fused distance (higher is better).
        """
        if device is None:
            device = torch.device("cpu")

        if isinstance(query_features, torch.Tensor):
            query_t = query_features.detach().to(device=device, dtype=torch.float32)
        else:
            query_features = np.asarray(query_features, dtype=np.float32)
            query_t = torch.from_numpy(query_features).to(device=device, dtype=torch.float32)

        if query_t.shape[0] == 0:
            return torch.empty((0, total_classes), device=device, dtype=torch.float32)

        query_dim = query_t.shape[1]
        classes = sorted(self.class_mu.keys())
        valid_classes = [cls for cls in classes if self.class_mu[cls].shape[0] == query_dim]

        # Start from very small logits for all classes.
        full_logits = torch.full(
            (query_t.shape[0], total_classes),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        if len(valid_classes) == 0:
            return full_logits

        self._ensure_predict_cache(device=device, query_dim=query_dim, valid_classes=valid_classes)

        q_norm = torch.nn.functional.normalize(query_t, p=2, dim=1)
        ncm_centers_t = self._predict_cache["ncm_centers_t"]      # [C, D]
        all_protos_t = self._predict_cache["all_protos_t"]        # [M, D]
        proto_labels_t = self._predict_cache["proto_labels_t"]    # [M]
        proto_class_index_t = self._predict_cache["proto_class_index_t"]  # [M]

        # Distances in cosine space.
        dist_ncm = 1.0 - torch.mm(q_norm, ncm_centers_t.t())      # [N, C]
        dist_proto_all = 1.0 - torch.mm(q_norm, all_protos_t.t()) # [N, M]

        C = len(valid_classes)
        N = query_t.shape[0]
        dist_sub = torch.full((N, C), float("inf"), device=device, dtype=dist_proto_all.dtype)
        if hasattr(dist_sub, "scatter_reduce_"):
            idx = proto_class_index_t.view(1, -1).expand(N, -1)
            dist_sub.scatter_reduce_(1, idx, dist_proto_all, reduce="amin", include_self=True)
        else:
            for i, cls in enumerate(valid_classes):
                mask = (proto_labels_t == cls)
                if mask.any():
                    min_d, _ = dist_proto_all[:, mask].min(dim=1)
                    dist_sub[:, i] = min_d
                else:
                    dist_sub[:, i] = dist_ncm[:, i]

        final_scores = self.alpha * dist_ncm + (1.0 - self.alpha) * dist_sub  # lower is better
        valid_class_ids = torch.tensor(valid_classes, device=device, dtype=torch.long)
        full_logits[:, valid_class_ids] = -final_scores
        return full_logits

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def get_class_prototypes_info(self, cls: int, k: int = 5) -> Tuple[np.ndarray, List[int]]:
        """Handle get class prototypes info."""
        if cls not in self.class_clusters or len(self.class_clusters[cls]) == 0:
            return np.zeros((0, 0)), []
            
        clusters = self.class_clusters[cls]
        sorted_clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
        top_clusters = sorted_clusters[:k]
        
        centers = np.stack([c.center for c in top_clusters], axis=0)  # [K, D]
        counts = [c.count for c in top_clusters]  # [K]
        
        return centers, counts

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def prototypes_per_class(self) -> Dict[int, int]:
        return {c: len(v) for c, v in self.class_clusters.items() if len(v) > 0}
