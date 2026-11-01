"""Train MF + BPR with the frozen split and negative-sampling protocol.

Run from the repository root:

    uv run python -m scripts.train_bpr --explicit-negative-proportion 0.0 --seed 42

Checkpoints and metrics are written under ``runs/``. Checkpoint selection uses
validation NDCG@20 only; test metrics are produced after training from the best
validation checkpoint.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from src.data.negatives import NegativeSampler
from src.data.processed import ProcessedSplits, load_processed_splits
from src.data.splitting import DEFAULT_BOUNDARIES, POSITIVE, TEST, VALIDATION
from src.evaluation.metrics import EvaluationSplit, evaluate
from src.models.bpr_mf import BPRMatrixFactorization
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


def _save_checkpoint(model: BPRMatrixFactorization, path: Path) -> None:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to runs/mf_bpr_p<percent>/seed_<seed>",
    )
    args = parser.parse_args()

    patience = int(args.patience)

    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if not 0.0 <= args.explicit_negative_proportion <= 1.0:
        raise ValueError("explicit negative proportion must be in [0, 1]")

    percent = round(args.explicit_negative_proportion * 100)
    experiment = f"mf_bpr_p{percent}"
    output_dir = args.output_dir or ROOT / "runs" / experiment / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_processed_splits(args.data_dir)
    split = _evaluation_split(splits)
    device = _resolve_device(args.device)

    torch.manual_seed(args.seed)
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
    model = BPRMatrixFactorization(
        train_users,
        splits.catalog_ids,
        embedding_dim=config.embedding_dim,
    )

    best_epoch = 0
    best_ndcg = float("-inf")
    stale_evaluations = 0
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
    test_metrics = evaluate(split, model, period=TEST)
    realized = sum(report.realized_explicit_proportion for report in history) / len(history)

    results: dict[str, Any] = {
        "experiment": experiment,
        "model": "mf_bpr",
        "seed": args.seed,
        "target_explicit_negative_proportion": args.explicit_negative_proportion,
        "realized_explicit_negative_proportion": realized,
        "split_boundaries": {
            "validation_start": DEFAULT_BOUNDARIES.validation_start.isoformat(),
            "test_start": DEFAULT_BOUNDARIES.test_start.isoformat(),
        },
        "training_config": asdict(config),
        "device": device,
        "best_epoch": best_epoch,
        "validation_ndcg@20_at_selection": best_ndcg,
        "epochs": [asdict(report) for report in history],
        "validation": validation_metrics,
        "test": test_metrics,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"wrote {checkpoint_path}")
    print(f"wrote {metrics_path}")
    print(
        f"seed={args.seed} realized_explicit={realized:.3f} "
        f"val_ndcg@20={validation_metrics['ndcg@20']:.6f} "
        f"test_ndcg@20={test_metrics['ndcg@20']:.6f}"
    )


if __name__ == "__main__":
    main()
