import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data_manager import DummyDataset


DEFAULT_FEATURE_CACHE_DIR = "/home/lyw/data/FeatureCache/LifeTopoDict"
FEATURE_CACHE_TRANSFORM_MODE = "test"
FEATURE_CACHE_EXTRACTION_MODE = "frozen_feature"
FEATURE_FILE_TEMPLATE = "{}_features.npy"
METADATA_FILE_TEMPLATE = "{}_metadata.json"


class FeatureCacheError(RuntimeError):
    """Raised when a requested feature cache is missing or incompatible."""


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "t", "yes", "y"):
            return True
        if lowered in ("0", "false", "f", "no", "n"):
            return False
    return bool(value)


def feature_cache_enabled(args):
    return _as_bool(args.get("use_feature_cache", False), default=False)


def feature_cache_strict(args):
    return _as_bool(args.get("feature_cache_strict", False), default=False)


def feature_cache_classifier_only(args):
    """Whether feature-cache runs should skip backbone construction entirely.

    In strict cache mode the cached features are the input contract, so the
    frozen backbone is not needed for classifier-only experiments. The option
    defaults to true under strict feature cache and can be disabled with
    ``feature_cache_skip_backbone=false`` for legacy profiling.
    """
    return (
        feature_cache_enabled(args)
        and feature_cache_strict(args)
        and _as_bool(args.get("feature_cache_skip_backbone", True), default=True)
    )


def feature_cache_classifier_torch_device(args, fallback_device):
    """Return the torch device for cached-feature classifier inference."""
    requested = str(args.get("feature_cache_classifier_device", "cpu")).strip().lower()
    if requested in ("cpu", "-1"):
        return torch.device("cpu")
    if requested in ("cuda", "gpu"):
        return fallback_device
    return torch.device(requested)


def safe_path_component(value):
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return text or "none"


def get_feature_cache_key(args):
    dataset = safe_path_component(args.get("dataset", "unknown_dataset"))
    backbone = safe_path_component(args.get("backbone_type", "unknown_backbone"))
    transform_mode = safe_path_component(
        args.get("feature_cache_transform_mode", FEATURE_CACHE_TRANSFORM_MODE)
    )
    extraction_mode = safe_path_component(
        args.get("feature_cache_extraction_mode", FEATURE_CACHE_EXTRACTION_MODE)
    )
    return "__".join(
        [
            "dataset-" + dataset,
            "backbone-" + backbone,
            "transform-" + transform_mode,
            "extract-" + extraction_mode,
        ]
    )


def get_feature_cache_root(args=None, cache_dir=None):
    args = args or {}
    root = cache_dir if cache_dir is not None else args.get("feature_cache_dir")
    if root is None or str(root).strip() == "":
        root = DEFAULT_FEATURE_CACHE_DIR
    return Path(os.path.expanduser(str(root))).resolve()


def get_feature_cache_dir(args, cache_dir=None):
    return get_feature_cache_root(args, cache_dir=cache_dir) / get_feature_cache_key(args)


def normalize_source(source):
    source = str(source).lower()
    if source not in ("train", "test"):
        raise ValueError("Unknown feature cache source {}.".format(source))
    return source


def get_feature_cache_paths(args, source, cache_dir=None):
    source = normalize_source(source)
    cache_path = get_feature_cache_dir(args, cache_dir=cache_dir)
    feature_path = cache_path / FEATURE_FILE_TEMPLATE.format(source)
    metadata_path = cache_path / METADATA_FILE_TEMPLATE.format(source)
    return feature_path, metadata_path


def build_transform_from_data_manager(data_manager, mode="test"):
    if mode == "train":
        trsf = transforms.Compose([*data_manager._train_trsf, *data_manager._common_trsf])
    elif mode == "flip":
        trsf = transforms.Compose(
            [
                *data_manager._test_trsf,
                transforms.RandomHorizontalFlip(p=1.0),
                *data_manager._common_trsf,
            ]
        )
    elif mode == "test":
        trsf = transforms.Compose([*data_manager._test_trsf, *data_manager._common_trsf])
    else:
        raise ValueError("Unknown mode {}.".format(mode))
    return trsf


def get_source_arrays(data_manager, source):
    source = normalize_source(source)
    if source == "train":
        return data_manager._train_data, data_manager._train_targets
    return data_manager._test_data, data_manager._test_targets


def get_full_split_dataset(data_manager, source, mode=FEATURE_CACHE_TRANSFORM_MODE):
    data, targets = get_source_arrays(data_manager, source)
    trsf = build_transform_from_data_manager(data_manager, mode=mode)
    return DummyDataset(data, targets, trsf, data_manager.use_path)


def _class_indices_array(indices):
    if isinstance(indices, np.ndarray):
        return indices.astype(np.int64, copy=False).ravel()
    return np.asarray(list(indices), dtype=np.int64).ravel()


def select_positions_by_class(targets, indices):
    targets = np.asarray(targets)
    class_indices = _class_indices_array(indices)
    selected = []
    for class_idx in class_indices:
        idxes = np.where(
            np.logical_and(targets >= int(class_idx), targets < int(class_idx) + 1)
        )[0]
        selected.append(idxes.astype(np.int64, copy=False))
    if not selected:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(selected).astype(np.int64, copy=False)


class CachedFeatureDataset(Dataset):
    """Dataset view over a full-split feature cache.

    The cache array remains in original DataManager split order. This dataset
    stores a class-selected position list so iteration order matches
    DataManager.get_dataset(indices, source, mode="test").
    """

    is_feature_cache = True

    def __init__(
        self,
        features,
        labels,
        original_indices,
        source,
        metadata=None,
        feature_path=None,
        metadata_path=None,
    ):
        assert len(labels) == len(original_indices), "Cached feature label size error."
        self.features = features
        self.labels = np.asarray(labels, dtype=np.int64)
        self.original_indices = np.asarray(original_indices, dtype=np.int64)
        self.source = normalize_source(source)
        self.metadata = metadata or {}
        self.feature_path = str(feature_path) if feature_path is not None else None
        self.metadata_path = str(metadata_path) if metadata_path is not None else None

    def __len__(self):
        return len(self.original_indices)

    def __getitem__(self, idx):
        source_idx = int(self.original_indices[idx])
        feature = torch.tensor(self.features[source_idx], dtype=torch.float32)
        label = int(self.labels[idx])
        return idx, feature, label


def _load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_cache_metadata(args, source, cache_dir=None):
    _, metadata_path = get_feature_cache_paths(args, source, cache_dir=cache_dir)
    if not metadata_path.exists():
        raise FeatureCacheError(
            "Feature cache metadata is missing for source={} at {}".format(
                source, metadata_path
            )
        )
    return _load_metadata(metadata_path)


def _validate_metadata(args, source, metadata):
    expected = {
        "dataset": args.get("dataset"),
        "backbone_type": args.get("backbone_type"),
        "source": normalize_source(source),
        "transform_mode": FEATURE_CACHE_TRANSFORM_MODE,
        "extraction_mode": FEATURE_CACHE_EXTRACTION_MODE,
    }
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value is not None and str(actual_value) != str(expected_value):
            mismatches.append((key, actual_value, expected_value))
    if mismatches:
        formatted = ", ".join(
            "{}={} (expected {})".format(key, actual, expected)
            for key, actual, expected in mismatches
        )
        raise FeatureCacheError("Feature cache metadata mismatch: {}".format(formatted))


def _missing_cache_error(args, source, feature_path, metadata_path):
    return FeatureCacheError(
        "Feature cache requested but source={} cache is missing for "
        "dataset={} backbone_type={} at {}. Expected files: {}, {}. "
        "Build it with scripts/build_feature_cache.py before running with "
        "feature_cache_strict=true.".format(
            source,
            args.get("dataset"),
            args.get("backbone_type"),
            feature_path.parent,
            feature_path.name,
            metadata_path.name,
        )
    )


def get_cached_feature_dataset(args, data_manager, indices, source):
    """Return a CachedFeatureDataset for the current class selection.

    Returns None when caching is disabled. If caching is enabled but unavailable,
    strict mode raises FeatureCacheError and non-strict mode logs a warning and
    returns None so callers can fall back to image datasets.
    """
    if not feature_cache_enabled(args):
        return None

    source = normalize_source(source)
    strict = feature_cache_strict(args)
    feature_path, metadata_path = get_feature_cache_paths(args, source)

    try:
        if not feature_path.exists() or not metadata_path.exists():
            raise _missing_cache_error(args, source, feature_path, metadata_path)

        metadata = _load_metadata(metadata_path)
        _validate_metadata(args, source, metadata)

        features = np.load(feature_path, mmap_mode="r")
        if features.ndim != 2:
            raise FeatureCacheError(
                "Feature cache {} must be 2-D, got shape {}.".format(
                    feature_path, features.shape
                )
            )

        _, source_targets = get_source_arrays(data_manager, source)
        if int(features.shape[0]) != int(len(source_targets)):
            raise FeatureCacheError(
                "Feature cache sample count mismatch for source={}: cache has {}, "
                "DataManager has {}.".format(source, features.shape[0], len(source_targets))
            )

        metadata_count = metadata.get("num_samples")
        if metadata_count is not None and int(metadata_count) != int(features.shape[0]):
            raise FeatureCacheError(
                "Feature cache metadata num_samples={} but feature array has {} rows.".format(
                    metadata_count, features.shape[0]
                )
            )

        positions = select_positions_by_class(source_targets, indices)
        labels = np.asarray(source_targets)[positions]
        dataset = CachedFeatureDataset(
            features=features,
            labels=labels,
            original_indices=positions,
            source=source,
            metadata=metadata,
            feature_path=feature_path,
            metadata_path=metadata_path,
        )
        logging.info(
            "[FeatureCache] Using cached {} features: samples={}, feature_dim={}, path={}".format(
                source, len(dataset), features.shape[1], feature_path
            )
        )
        return dataset
    except FeatureCacheError as exc:
        if strict:
            raise
        logging.warning("[FeatureCache] {} Falling back to image dataset.".format(exc))
        return None


def get_cached_feature_dataset_split(
    args,
    data_manager,
    indices,
    source,
    val_samples_per_class=0,
):
    """Return train/val cached-feature datasets with a per-class holdout split.

    The returned datasets share the same backing memory-mapped feature array;
    only the selected row positions differ. When feature cache is disabled or
    unavailable, ``(None, None)`` is returned so callers can fall back to the
    image-backed split path.
    """
    if not feature_cache_enabled(args):
        return None, None

    source = normalize_source(source)
    strict = feature_cache_strict(args)
    feature_path, metadata_path = get_feature_cache_paths(args, source)

    try:
        if not feature_path.exists() or not metadata_path.exists():
            raise _missing_cache_error(args, source, feature_path, metadata_path)

        metadata = _load_metadata(metadata_path)
        _validate_metadata(args, source, metadata)

        features = np.load(feature_path, mmap_mode="r")
        if features.ndim != 2:
            raise FeatureCacheError(
                "Feature cache {} must be 2-D, got shape {}.".format(
                    feature_path, features.shape
                )
            )

        _, source_targets = get_source_arrays(data_manager, source)
        if int(features.shape[0]) != int(len(source_targets)):
            raise FeatureCacheError(
                "Feature cache sample count mismatch for source={}: cache has {}, "
                "DataManager has {}.".format(source, features.shape[0], len(source_targets))
            )

        metadata_count = metadata.get("num_samples")
        if metadata_count is not None and int(metadata_count) != int(features.shape[0]):
            raise FeatureCacheError(
                "Feature cache metadata num_samples={} but feature array has {} rows.".format(
                    metadata_count, features.shape[0]
                )
            )

        source_targets = np.asarray(source_targets, dtype=np.int64)
        class_indices = _class_indices_array(indices)
        train_positions = []
        val_positions = []

        for class_idx in class_indices:
            class_positions = np.where(
                np.logical_and(source_targets >= int(class_idx), source_targets < int(class_idx) + 1)
            )[0]
            if class_positions.size == 0:
                continue

            if val_samples_per_class is None or int(val_samples_per_class) <= 0:
                val_count = 0
            else:
                val_count = min(int(val_samples_per_class), max(0, int(class_positions.size) - 1))

            if val_count > 0:
                val_choice = np.random.choice(class_positions, size=val_count, replace=False)
                val_choice = np.sort(val_choice.astype(np.int64, copy=False))
                train_choice = np.setdiff1d(class_positions, val_choice, assume_unique=False)
            else:
                val_choice = np.empty((0,), dtype=np.int64)
                train_choice = class_positions

            train_positions.append(np.sort(train_choice.astype(np.int64, copy=False)))
            if val_choice.size > 0:
                val_positions.append(val_choice)

        if len(train_positions) == 0:
            raise FeatureCacheError(
                "Feature cache split for source={} and indices={} produced no train samples.".format(
                    source, list(class_indices)
                )
            )

        train_positions = np.concatenate(train_positions).astype(np.int64, copy=False)
        train_labels = source_targets[train_positions]
        train_dataset = CachedFeatureDataset(
            features=features,
            labels=train_labels,
            original_indices=train_positions,
            source=source,
            metadata=metadata,
            feature_path=feature_path,
            metadata_path=metadata_path,
        )

        if len(val_positions) == 0:
            val_dataset = None
        else:
            val_positions = np.concatenate(val_positions).astype(np.int64, copy=False)
            val_labels = source_targets[val_positions]
            val_dataset = CachedFeatureDataset(
                features=features,
                labels=val_labels,
                original_indices=val_positions,
                source=source,
                metadata=metadata,
                feature_path=feature_path,
                metadata_path=metadata_path,
            )

        logging.info(
            "[FeatureCache] Split cached {} features: train_samples={}, val_samples={}, path={}".format(
                source,
                len(train_dataset),
                0 if val_dataset is None else len(val_dataset),
                feature_path,
            )
        )
        return train_dataset, val_dataset
    except FeatureCacheError as exc:
        if strict:
            raise
        logging.warning("[FeatureCache] {} Falling back to image dataset split.".format(exc))
        return None, None


def is_cached_feature_dataset(dataset):
    if getattr(dataset, "is_feature_cache", False):
        return True
    children = getattr(dataset, "datasets", None)
    if children is not None:
        return all(is_cached_feature_dataset(child) for child in children)
    child = getattr(dataset, "dataset", None)
    if child is not None and child is not dataset:
        return is_cached_feature_dataset(child)
    return False


def is_cached_feature_loader(loader):
    dataset = getattr(loader, "dataset", None)
    return dataset is not None and is_cached_feature_dataset(dataset)


def log_feature_cache_loader(loader, name):
    dataset = getattr(loader, "dataset", None)
    if is_cached_feature_loader(loader):
        logging.info(
            "[FeatureCache] {} is backed by cached features: source={}, samples={}, path={}".format(
                name,
                getattr(dataset, "source", "unknown"),
                len(dataset),
                getattr(dataset, "feature_path", "unknown"),
            )
        )
    else:
        logging.info("[FeatureCache] {} uses image dataset/backbone path.".format(name))


def build_feature_cache_metadata(
    args,
    source,
    num_samples,
    feature_dim,
    dtype,
    config_path=None,
    batch_size=None,
    num_workers=None,
):
    source = normalize_source(source)
    metadata = {
        "dataset": str(args.get("dataset")),
        "backbone_type": str(args.get("backbone_type")),
        "source": source,
        "num_samples": int(num_samples),
        "feature_dim": int(feature_dim),
        "dtype": np.dtype(dtype).name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path) if config_path is not None else None,
        "transform_mode": FEATURE_CACHE_TRANSFORM_MODE,
        "extraction_mode": FEATURE_CACHE_EXTRACTION_MODE,
        "cache_key": get_feature_cache_key(args),
        "batch_size": int(batch_size) if batch_size is not None else None,
        "num_workers": int(num_workers) if num_workers is not None else None,
    }
    if args.get("feature_cache_dtype") is not None:
        metadata["requested_feature_cache_dtype"] = str(args.get("feature_cache_dtype"))
    return metadata


def write_feature_cache_metadata(metadata_path, metadata):
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, metadata_path)
