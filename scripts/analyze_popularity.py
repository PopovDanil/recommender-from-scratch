"""Measure train-popularity exposure and recall from exported test rankings."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from src.data.splitting import POSITIVE

# Log-like fixed thresholds, independent of validation and test results.
BINS = ("1-9", "10-99", "100-999", "1000+")


def popularity_bin(count: int) -> str:
    if count < 1:
        raise ValueError("Catalog popularity must be positive")
    return BINS[0 if count < 10 else 1 if count < 100 else 2 if count < 1000 else 3]


def summarize(rows: list[dict[str, Any]], counts: dict[int, int]) -> dict[str, Any]:
    users = [row["user_id"] for row in rows]
    if len(users) != len(set(users)) or not users:
        raise ValueError("Ranking export must contain unique eligible users")
    result: dict[str, Any] = {}
    for name in BINS:
        exposures = []
        recalls = []
        total_targets = 0
        for row in rows:
            top = row["top20"]
            if not top:
                raise ValueError("Empty candidate ranking")
            exposures.append(
                sum(popularity_bin(counts[item["movie_id"]]) == name for item in top) / len(top)
            )
            targets = [
                item
                for item in row["test_items"]
                if item["test_label"] == POSITIVE
                and popularity_bin(counts[item["movie_id"]]) == name
            ]
            if targets:
                total_targets += len(targets)
                recalls.append(sum(item["rank"] <= 20 for item in targets) / len(targets))
        result[name] = {
            "catalog_items": sum(popularity_bin(c) == name for c in counts.values()),
            "eligible_users": len(recalls),
            "relevant_test_targets": total_targets,
            "exposure@20": sum(exposures) / len(exposures),
            "recall@20": sum(recalls) / len(recalls) if recalls else None,
        }
    return result


def signature(rows: list[dict[str, Any]]) -> dict[int, tuple[int, str, tuple[int, ...]]]:
    return {
        row["user_id"]: (
            row["candidate_set_size"],
            row["candidate_sha256"],
            tuple(
                sorted(
                    item["movie_id"] for item in row["test_items"] if item["test_label"] == POSITIVE
                )
            ),
        )
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rankings", nargs="+", type=Path)
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/ml-1m/catalog.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = {
        int(i): int(n)
        for i, n in pd.read_csv(args.catalog)[["movie_id", "train_positive_count"]].itertuples(
            index=False, name=None
        )
    }
    reference = None
    results = []
    for path in args.rankings:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        current = signature(rows)
        if len(current) != 773:
            raise ValueError(f"{path}: expected 773 users, got {len(current)}")
        if reference is not None and current != reference:
            raise ValueError(f"{path}: user, target or candidate population differs")
        reference = current
        results.append(
            {
                "source": str(path),
                "configuration": rows[0]["configuration"],
                "seed": rows[0]["seed"],
                "regularization_on": rows[0]["regularization_on"],
                "bins": summarize(rows, counts),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"bins": list(BINS), "runs": results}, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
