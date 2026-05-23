#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_splits(values):
    splits = []
    for value in values:
        split = str(value).lower()
        if split not in ("train", "test"):
            raise argparse.ArgumentTypeError(
                "Unknown split {}. Expected train or test.".format(value)
            )
        if split not in splits:
            splits.append(split)
    return splits


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Build frozen-backbone feature caches for LifeTopoDict learners."
    )
    parser.add_argument("--config", type=str, required=True, help="JSON config path.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for extraction. Use 0 for cuda:0 or cpu/-1 for CPU.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Feature cache root directory. Defaults to /home/lyw/data/FeatureCache/LifeTopoDict.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Numpy dtype used on disk.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=("train", "test"),
        help="Splits to cache.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature arrays and metadata.",
    )
    return parser


def load_json(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _device_from_arg(device_arg):
    if str(device_arg).lower() == "cpu" or str(device_arg) == "-1":
        return torch.device("cpu")
    if str(device_arg).startswith("cuda:"):
        return torch.device(device_arg)
    return torch.device("cuda:{}".format(device_arg))


def _normalise_args(config, cli_args):
    args = dict(config)
    args["device"] = [_device_from_arg(cli_args.device)]
    if cli_args.cache_dir is not None:
        args["feature_cache_dir"] = cli_args.cache_dir
    else:
        args.setdefault("feature_cache_dir", None)
    args["feature_cache_dtype"] = cli_args.dtype
    args.setdefault("model_name", "life_topo_dict")
    return args


def _set_random(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _extract_split_features(model, loader, device, dtype, output_path, num_samples):
    mmap_array = None
    feature_dim = None
    seen = 0
    np_dtype = np.dtype(dtype)
    model.eval()
    with torch.no_grad():
        for batch_indices, inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            batch_features = model.extract_vector(inputs)
            if isinstance(batch_features, dict):
                batch_features = batch_features["features"]
            batch_features = batch_features.detach().cpu().numpy()
            if batch_features.ndim != 2:
                raise RuntimeError(
                    "Expected 2-D features, got shape {}.".format(batch_features.shape)
                )

            if mmap_array is None:
                feature_dim = int(batch_features.shape[1])
                mmap_array = np.lib.format.open_memmap(
                    str(output_path),
                    mode="w+",
                    dtype=np_dtype,
                    shape=(int(num_samples), feature_dim),
                )

            batch_indices = batch_indices.numpy().astype(np.int64, copy=False)
            mmap_array[batch_indices] = batch_features.astype(np_dtype, copy=False)
            seen += int(batch_features.shape[0])

    if mmap_array is None:
        raise RuntimeError("No samples found while extracting features.")

    mmap_array.flush()
    del mmap_array
    if seen != int(num_samples):
        raise RuntimeError(
            "Extracted {} samples but expected {}.".format(seen, num_samples)
        )
    return seen, feature_dim, np_dtype.name


def _save_split(args, data_manager, model, source, cli_args, device):
    from utils.feature_cache import (
        FEATURE_CACHE_TRANSFORM_MODE,
        build_feature_cache_metadata,
        get_feature_cache_paths,
        get_full_split_dataset,
        write_feature_cache_metadata,
    )

    feature_path, metadata_path = get_feature_cache_paths(
        args, source, cache_dir=cli_args.cache_dir
    )
    if feature_path.exists() and metadata_path.exists() and not cli_args.overwrite:
        logging.info(
            "Skipping existing {} cache: {} (use --overwrite to rebuild)".format(
                source, feature_path
            )
        )
        return
    if (feature_path.exists() or metadata_path.exists()) and not cli_args.overwrite:
        raise RuntimeError(
            "Partial {} cache exists at {}. Use --overwrite to rebuild.".format(
                source, feature_path.parent
            )
        )

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = get_full_split_dataset(
        data_manager, source, mode=FEATURE_CACHE_TRANSFORM_MODE
    )
    loader = DataLoader(
        dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
        pin_memory=device.type == "cuda",
    )

    logging.info(
        "Extracting {} split: samples={}, batch_size={}, device={}".format(
            source, len(dataset), cli_args.batch_size, device
        )
    )
    tmp_path = feature_path.with_suffix(feature_path.suffix + ".tmp")
    seen, feature_dim, saved_dtype = _extract_split_features(
        model,
        loader,
        device=device,
        dtype=cli_args.dtype,
        output_path=tmp_path,
        num_samples=len(dataset),
    )
    os.replace(tmp_path, feature_path)

    metadata = build_feature_cache_metadata(
        args=args,
        source=source,
        num_samples=seen,
        feature_dim=feature_dim,
        dtype=saved_dtype,
        config_path=cli_args.config,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
    )
    write_feature_cache_metadata(metadata_path, metadata)
    logging.info(
        "Saved {} cache: features={}, metadata={}, shape={}, dtype={}".format(
            source, feature_path, metadata_path, (seen, feature_dim), saved_dtype
        )
    )


def main():
    cli_args = setup_parser().parse_args()
    cli_args.splits = _parse_splits(cli_args.splits)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [build_feature_cache.py] => %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    from utils.data_manager import DataManager
    from utils.inc_net import SimpleVitNetKNN
    from utils.feature_cache import get_feature_cache_dir

    config = load_json(cli_args.config)
    args = _normalise_args(config, cli_args)

    seed = args.get("seed", [0])
    if isinstance(seed, list):
        seed = seed[0]
    args["seed"] = int(seed)
    _set_random(args["seed"])

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    device = args["device"][0]
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")

    logging.info("Feature cache directory: {}".format(get_feature_cache_dir(args)))
    logging.info(
        "Building cache for dataset={}, backbone_type={}, splits={}, dtype={}".format(
            args["dataset"], args["backbone_type"], cli_args.splits, cli_args.dtype
        )
    )

    model = SimpleVitNetKNN(args, True).to(device)
    model.eval()

    for source in cli_args.splits:
        _save_split(args, data_manager, model, source, cli_args, device)


if __name__ == "__main__":
    main()
