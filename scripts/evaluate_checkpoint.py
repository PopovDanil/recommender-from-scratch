"""Explicitly evaluate a selected validation checkpoint on the frozen test set."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from src.data.processed import ProcessedSplits, load_processed_splits
from src.data.splitting import POSITIVE, TEST
from src.evaluation.metrics import DEFAULT_METRICS, evaluate_per_user
from src.models.bpr_mf import BPRMatrixFactorization
from src.models.lightgcn import LightGCN
from src.training.artifacts import data_fingerprint, sha256, write_json
from src.training.bpr import BPRTrainingConfig

from scripts.train import ROOT, _build_model, _evaluation_split, _per_user_payload


def load_model(
    run_dir: Path, splits: ProcessedSplits, metadata: dict[str, Any], device: str = "cpu"
) -> BPRMatrixFactorization | LightGCN:
    config = BPRTrainingConfig(**metadata["training_config"])
    model_config = metadata["model_config"]
    name = metadata["model"]
    regularization_on = metadata.get(
        "regularization_on", "initial" if name == "mf_bpr" else "propagated"
    )
    users = sorted(
        int(user)
        for user in splits.train.loc[splits.train["label"] == POSITIVE, "user_id"].unique()
    )
    model = _build_model(
        name,
        users,
        splits,
        config,
        model_config.get("n_layers") or 2,
        model_config.get("alpha") if model_config.get("alpha") is not None else 0.6,
        model_config.get("beta") if model_config.get("beta") is not None else -0.1,
        model_config.get("gamma") if model_config.get("gamma") is not None else 0.2,
        regularization_on,
    )
    path = run_dir / "best.pt"
    if metadata.get("checkpoint_sha256") and sha256(path) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return model


def verify_data(metadata: dict[str, Any], data_dir: Path, manifest: Path) -> None:
    if data_fingerprint(data_dir) != metadata.get("data_sha256"):
        raise ValueError("Processed data checksum mismatch or missing provenance")
    if sha256(manifest) != metadata.get("manifest_sha256"):
        raise ValueError("Manifest checksum mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml-1m")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    path = args.run_dir / "metrics.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if "test" in metadata or "test_per_user" in metadata:
        raise ValueError("Test evaluation already exists for this run")
    verify_data(metadata, args.data_dir, args.manifest)
    splits = load_processed_splits(args.data_dir)
    model = load_model(args.run_dir, splits, metadata, args.device)
    per_user = evaluate_per_user(_evaluation_split(splits), model, period=TEST)
    if not per_user:
        raise ValueError("No eligible test users")
    metadata["test"] = {
        m.name: sum(v[m.name] for v in per_user.values()) / len(per_user) for m in DEFAULT_METRICS
    }
    metadata["test_per_user"] = _per_user_payload(per_user)
    metadata["test_device"] = args.device
    write_json(path, metadata)
    print(f"wrote test metrics to {path}")


if __name__ == "__main__":
    main()
