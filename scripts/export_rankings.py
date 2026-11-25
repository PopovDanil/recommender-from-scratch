"""Export canonical top-20 rankings and the full ranks of eligible test items."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.loading import load_movies
from src.data.processed import ProcessedSplits, load_processed_splits
from src.data.splitting import EXPLICIT_NEGATIVE, TEST
from src.evaluation.metrics import iter_rankings
from src.models.base import Recommender
from src.models.popularity import Popularity

from scripts.evaluate_checkpoint import load_model, verify_data
from scripts.train import ROOT, _evaluation_split


def export_rows(
    splits: ProcessedSplits, model: Recommender, titles: dict[int, str], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    counts = {
        int(i): int(n)
        for i, n in splits.catalog[["movie_id", "train_positive_count"]].itertuples(
            index=False, name=None
        )
    }
    observed = {
        (int(u), int(i)): (int(r), str(label))
        for u, i, r, label in splits.test[["user_id", "movie_id", "rating", "label"]].itertuples(
            index=False, name=None
        )
    }
    rows = []
    for user, ranking, relevant in iter_rankings(_evaluation_split(splits), model, TEST):
        top = []
        targets = []
        for rank, (item, score) in enumerate(ranking, 1):
            rating, label = observed.get((user, item), (None, "unobserved"))
            if rank > 20 and item not in relevant and label != EXPLICIT_NEGATIVE:
                continue
            entry = {
                "movie_id": item,
                "title": titles[item],
                "rank": rank,
                "score": score,
                "test_rating": rating,
                "test_label": label,
                "train_positive_count": counts[item],
            }
            if rank <= 20:
                top.append(entry)
            if item in relevant or label == EXPLICIT_NEGATIVE:
                targets.append(entry)
        rows.append(
            {
                "user_id": user,
                "configuration": metadata["experiment"],
                "seed": metadata.get("seed"),
                "checkpoint_epoch": metadata.get("best_epoch"),
                "regularization_on": metadata.get("regularization_on", "not_applicable"),
                "candidate_set_size": len(ranking),
                "candidate_sha256": hashlib.sha256(
                    ",".join(str(i) for i in sorted(item for item, _ in ranking)).encode()
                ).hexdigest(),
                "top20": top,
                "test_items": targets,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--popularity", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml-1m")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.yaml")
    parser.add_argument("--movies", type=Path, default=ROOT / "data/ml-1m/movies.dat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy", action="store_true", help="Allow unverified old checkpoints"
    )
    args = parser.parse_args()
    splits = load_processed_splits(args.data_dir)
    if args.popularity:
        model: Recommender = Popularity.from_train(splits.train, splits.catalog_ids)
        metadata: dict[str, Any] = {
            "experiment": "popularity",
            "regularization_on": "not_applicable",
        }
    else:
        metadata = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
        if metadata.get("data_sha256"):
            verify_data(metadata, args.data_dir, args.manifest)
            if "test" not in metadata:
                raise ValueError("Select and explicitly evaluate the checkpoint before export")
        elif not args.allow_legacy:
            raise ValueError(
                "Legacy checkpoint has no data fingerprint; use --allow-legacy explicitly"
            )
        else:
            metadata["regularization_on"] = (
                "initial" if metadata["model"] == "mf_bpr" else "propagated"
            )
            print("Warning: legacy checkpoint provenance is not verified")
        model = load_model(args.run_dir, splits, metadata)
    titles = {
        int(i): str(t)
        for i, t in load_movies(args.movies)[["movie_id", "title"]].itertuples(
            index=False, name=None
        )
    }
    rows = export_rows(splits, model, titles, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"wrote {len(rows)} users to {args.output}")


if __name__ == "__main__":
    main()
