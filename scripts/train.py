"""Train a BPR recommender with the frozen split and negative-sampling protocol.

Run from the repository root:

    uv run python -m scripts.train --model mf_bpr \
        --explicit-negative-proportion 0.0 --seed 42
    uv run python -m scripts.train --model lightgcn_bpr \
        --explicit-negative-proportion 0.0 --seed 42
    uv run python -m scripts.train --model lightgcnpp_bpr \
        --explicit-negative-proportion 0.0 --seed 42

Checkpoints, per-epoch metrics and per-user validation/test metrics are written
under ``runs/``. Checkpoint selection uses validation NDCG@20 only; test metrics
require an explicit scripts.evaluate_checkpoint invocation.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from src.data.negatives import NegativeSampler
from src.data.processed import ProcessedSplits, load_processed_splits
from src.data.splitting import DEFAULT_BOUNDARIES, POSITIVE, VALIDATION
from src.evaluation.metrics import (
    EvaluationSplit,
    evaluate,
    evaluate_per_user,
)
from src.experiments import MODELS
from src.models.bpr_mf import BPRMatrixFactorization
from src.models.lightgcn import LightGCN, positive_interaction_edges
from src.models.lightgcnpp import LightGCNPlusPlus
from src.training.artifacts import (
    data_fingerprint,
    git_commit,
    reserve_run,
    run_id,
    sha256,
    write_json,
)
from src.training.bpr import BPRTrainingConfig, EpochReport, train_bpr

ROOT = Path(__file__).resolve().parents[1]


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _evaluation_split(splits: ProcessedSplits) -> EvaluationSplit:
    return EvaluationSplit(
        train=splits.train,
        validation=splits.validation,
        test=splits.test,
        catalog=splits.catalog_ids,
    )


def _build_model(
    name: str,
    train_users: list[int],
    splits: ProcessedSplits,
    config: BPRTrainingConfig,
    n_layers: int,
    alpha: float,
    beta: float,
    gamma: float,
    regularization_on: str = "initial",
) -> BPRMatrixFactorization | LightGCN:
    if name == "mf_bpr":
        return BPRMatrixFactorization(
            train_users,
            splits.catalog_ids,
            embedding_dim=config.embedding_dim,
        )

    user_edges, item_edges = positive_interaction_edges(splits.train, splits.catalog_ids)
    if name == "lightgcnpp_bpr":
        return LightGCNPlusPlus(
            train_users,
            splits.catalog_ids,
            user_edges,
            item_edges,
            n_layers=n_layers,
            embedding_dim=config.embedding_dim,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            regularization_on=regularization_on,
        )

    return LightGCN(
        train_users,
        splits.catalog_ids,
        user_edges,
        item_edges,
        n_layers=n_layers,
        embedding_dim=config.embedding_dim,
        regularization_on=regularization_on,
    )


def _save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, path)


def _per_user_payload(per_user: dict[int, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        str(user_id): {name: float(value) for name, value in values.items()}
        for user_id, values in per_user.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="mf_bpr")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ml-1m",
        help="directory with train.csv, validation.csv, test.csv and catalog.csv",
    )
    parser.add_argument(
        "--explicit-negative-proportion",
        type=float,
        default=0.0,
        help="target share of explicit training negatives in each epoch",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2, help="LightGCN propagation layers")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="LightGCN++ left degree exponent",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=-0.1,
        help="LightGCN++ right degree exponent",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="LightGCN++ weight on the input layer",
    )
    parser.add_argument("--regularization-on", choices=("initial", "propagated"), default="initial")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.yaml")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to a configuration-hashed directory under runs/",
    )
    args = parser.parse_args()

    patience = int(args.patience)

    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if not 0.0 <= args.explicit_negative_proportion <= 1.0:
        raise ValueError("explicit negative proportion must be in [0, 1]")

    percent = round(args.explicit_negative_proportion * 100)
    experiment = f"{args.model}_p{percent}"
    if args.model == "mf_bpr" and args.regularization_on != "initial":
        raise ValueError("MF supports initial-embedding regularization only")
    if args.regularization_on == "propagated":
        experiment = f"{args.model}_legacy_p{percent}"
    if patience <= 0:
        raise ValueError("patience must be positive")
    splits = load_processed_splits(args.data_dir)
    split = _evaluation_split(splits)
    device = _resolve_device(args.device)
    cli = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "output_dir"
    }
    fingerprint = data_fingerprint(args.data_dir)
    manifest_sha = sha256(args.manifest)
    commit = git_commit(ROOT)
    identity = run_id(
        {
            "cli": cli,
            "device": device,
            "data": fingerprint,
            "manifest": manifest_sha,
            "commit": commit,
        }
    )
    output_dir = args.output_dir or ROOT / "runs" / experiment / f"seed_{args.seed}_{identity}"
    reserve_run(output_dir)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sampler = NegativeSampler(
        splits.train,
        splits.catalog_ids,
        args.explicit_negative_proportion,
        seed=args.seed,
    )
    train_users = sorted(
        {
            int(user)
            for user in splits.train.loc[splits.train["label"] == POSITIVE, "user_id"].tolist()
        }
    )
    config = BPRTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
    )
    model = _build_model(
        args.model,
        train_users,
        splits,
        config,
        args.n_layers,
        args.alpha,
        args.beta,
        args.gamma,
        args.regularization_on,
    )

    best_epoch = 0
    best_ndcg = float("-inf")
    stale_evaluations = 0
    validation_history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "best.pt"

    def on_epoch_end(report: EpochReport) -> bool:
        nonlocal best_epoch, best_ndcg, stale_evaluations

        message = (
            f"epoch {report.epoch:3d} loss={report.loss:.5f} "
            f"realized_explicit={report.realized_explicit_proportion:.3f}"
        )
        should_evaluate = report.epoch % args.eval_every == 0 or report.epoch == config.epochs
        if not should_evaluate:
            print(message)
            return False

        model.eval()
        validation = evaluate(
            split,
            model,
            period=VALIDATION,
            metrics=["ndcg@20"],
        )
        value = validation["ndcg@20"]
        validation_history.append({"epoch": report.epoch, "ndcg@20": value})
        improved = value > best_ndcg
        if improved:
            best_ndcg = value
            best_epoch = report.epoch
            stale_evaluations = 0
            _save_checkpoint(model, checkpoint_path)
        else:
            stale_evaluations += 1

        print(f"{message} val_ndcg@20={value:.6f} best_epoch={best_epoch}")
        return stale_evaluations >= patience

    history = train_bpr(
        model,
        sampler,
        config,
        device=device,
        seed=args.seed,
        on_epoch_end=on_epoch_end,
    )

    if best_epoch == 0:
        raise RuntimeError("training did not produce a validation checkpoint")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint)
    model.eval()

    validation_metrics = evaluate(split, model, period=VALIDATION)
    validation_per_user = evaluate_per_user(split, model, period=VALIDATION)
    realized = sum(report.realized_explicit_proportion for report in history) / len(history)

    results: dict[str, Any] = {
        "experiment": experiment,
        "model": args.model,
        "seed": args.seed,
        "target_explicit_negative_proportion": args.explicit_negative_proportion,
        "realized_explicit_negative_proportion": realized,
        "split_boundaries": {
            "validation_start": DEFAULT_BOUNDARIES.validation_start.isoformat(),
            "test_start": DEFAULT_BOUNDARIES.test_start.isoformat(),
        },
        "training_config": asdict(config),
        "model_config": {
            "embedding_dim": args.embedding_dim,
            "n_layers": args.n_layers if args.model.startswith("lightgcn") else None,
            "alpha": args.alpha if args.model == "lightgcnpp_bpr" else None,
            "beta": args.beta if args.model == "lightgcnpp_bpr" else None,
            "gamma": args.gamma if args.model == "lightgcnpp_bpr" else None,
        },
        "device": device,
        "cli": cli,
        "run_id": identity,
        "git_commit": commit,
        "manifest_sha256": manifest_sha,
        "data_sha256": fingerprint,
        "checkpoint_sha256": sha256(checkpoint_path),
        "regularization_on": args.regularization_on,
        "validation_history": validation_history,
        "best_epoch": best_epoch,
        "validation_ndcg@20_at_selection": best_ndcg,
        "epochs": [asdict(report) for report in history],
        "validation": validation_metrics,
        "validation_per_user": _per_user_payload(validation_per_user),
    }
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, results)

    print(f"wrote {checkpoint_path}")
    print(f"wrote {metrics_path}")
    print(
        f"seed={args.seed} realized_explicit={realized:.3f} "
        f"val_ndcg@20={validation_metrics['ndcg@20']:.6f}"
    )


if __name__ == "__main__":
    main()
