"""Evaluate the most-popular baseline on the frozen validation and test splits."""

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.processed import load_processed_splits
from src.data.splitting import DEFAULT_BOUNDARIES, TEST, VALIDATION
from src.evaluation.metrics import EvaluationSplit, evaluate
from src.models.popularity import Popularity

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ml-1m",
        help="directory with train.csv, validation.csv, test.csv and catalog.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "popularity" / "metrics.json",
        help="path for the JSON metrics report",
    )
    args = parser.parse_args()

    splits = load_processed_splits(args.data_dir)
    split = EvaluationSplit(
        train=splits.train,
        validation=splits.validation,
        test=splits.test,
        catalog=splits.catalog_ids,
    )
    model = Popularity.from_train(splits.train, splits.catalog_ids)

    results: dict[str, Any] = {
        "experiment": "popularity",
        "model": "popularity",
        "split_boundaries": {
            "validation_start": DEFAULT_BOUNDARIES.validation_start.isoformat(),
            "test_start": DEFAULT_BOUNDARIES.test_start.isoformat(),
        },
        "validation": evaluate(split, model, period=VALIDATION),
        "test": evaluate(split, model, period=TEST),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"validation ndcg@20={results['validation']['ndcg@20']:.6f}")
    print(f"test ndcg@20={results['test']['ndcg@20']:.6f}")


if __name__ == "__main__":
    main()
