"""Prepare the frozen MovieLens-1M splits and their YAML manifest.

Run from the repository root:

    uv run python -m scripts.prepare_data

The script writes CSV splits and the catalog under ``data/processed/ml-1m/`` and
a tracked manifest at ``data/manifest.yaml``. It is deterministic for the fixed
release and should be rerun through the protocol, not edited by hand.
"""

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd
import yaml
from src.data.audit import full_audit
from src.data.loading import load_movies, load_ratings
from src.data.splitting import (
    DEFAULT_BOUNDARIES,
    SPLITS,
    TRAIN,
    build_catalog,
    prepare_ratings,
)
from src.experiments import (
    EXPLICIT_NEGATIVE_PROPORTIONS,
    MODELS,
    SEEDS,
    experiment_grid,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL: Final = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SPLIT_COLUMNS: Final = ("user_id", "movie_id", "rating", "timestamp", "label")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat().replace("+00:00", "Z")


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _write_split(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    ordered = frame.sort_values(["timestamp", "user_id", "movie_id"]).loc[:, list(SPLIT_COLUMNS)]
    ordered.to_csv(path, index=False)
    return {
        "path": _relative(path),
        "rows": len(ordered),
        "sha256": _sha256(path),
    }


def _write_catalog(catalog: pd.DataFrame, path: Path) -> dict[str, object]:
    catalog.to_csv(path, index=False)
    return {
        "path": _relative(path),
        "rows": len(catalog),
        "sha256": _sha256(path),
    }


def _dataset_section(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    data_dir: Path,
) -> dict[str, object]:
    archive = data_dir.parent / "ml-1m.zip"
    rating_counts = {
        str(rating): int(count)
        for rating, count in ratings["rating"].value_counts().sort_index().items()
    }
    raw_files = {
        name: _sha256(data_dir / name) for name in ("ratings.dat", "users.dat", "movies.dat")
    }

    section: dict[str, object] = {
        "name": "MovieLens-1M",
        "source": SOURCE_URL,
        "n_ratings": len(ratings),
        "n_users": int(ratings["user_id"].nunique()),
        "n_movies_rated": int(ratings["movie_id"].nunique()),
        "n_movies_in_release": len(movies),
        "rating_counts": rating_counts,
        "timestamp_range": {
            "min": _iso(int(ratings["timestamp"].min())),
            "max": _iso(int(ratings["timestamp"].max())),
        },
        "raw_files": raw_files,
    }
    if archive.exists():
        section["archive"] = {
            "path": _relative(archive),
            "sha256": _sha256(archive),
        }
    return section


def _split_section(ratings: pd.DataFrame) -> dict[str, object]:
    return {
        "strategy": "global_timestamp_boundaries",
        "timezone": "UTC",
        "validation_start": DEFAULT_BOUNDARIES.validation_start.isoformat().replace("+00:00", "Z"),
        "test_start": DEFAULT_BOUNDARIES.test_start.isoformat().replace("+00:00", "Z"),
        "validation_start_epoch": DEFAULT_BOUNDARIES.validation_epoch,
        "test_start_epoch": DEFAULT_BOUNDARIES.test_epoch,
        "assignment": (
            "timestamp < validation_start -> train; "
            "validation_start <= timestamp < test_start -> validation; "
            "timestamp >= test_start -> test"
        ),
        "tie_policy": "ratings with identical timestamps stay in the same split",
        "ratings_at_boundaries": {
            "validation_start": int(
                (ratings["timestamp"] == DEFAULT_BOUNDARIES.validation_epoch).sum()
            ),
            "test_start": int((ratings["timestamp"] == DEFAULT_BOUNDARIES.test_epoch).sum()),
        },
    }


def _experiment_section() -> dict[str, object]:
    return {
        "models": list(MODELS),
        "target_explicit_negative_proportions": list(EXPLICIT_NEGATIVE_PROPORTIONS),
        "seeds": list(SEEDS),
        "configurations": [experiment.as_dict() for experiment in experiment_grid()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "ml-1m",
        help="directory containing the raw MovieLens-1M files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ml-1m",
        help="directory for generated splits and catalog",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifest.yaml",
        help="path for the tracked YAML manifest",
    )
    args = parser.parse_args()

    ratings = load_ratings(args.data_dir / "ratings.dat")
    movies = load_movies(args.data_dir / "movies.dat")
    prepared = prepare_ratings(ratings)
    catalog = build_catalog(prepared[prepared["split"] == TRAIN])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_artifacts = {
        split: _write_split(
            prepared[prepared["split"] == split],
            args.output_dir / f"{split}.csv",
        )
        for split in SPLITS
    }
    catalog_artifact = _write_catalog(catalog, args.output_dir / "catalog.csv")

    audit = full_audit(prepared, catalog)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": _dataset_section(ratings, movies, args.data_dir),
        "split": _split_section(ratings),
        "periods": audit["periods"],
        "catalog": {
            "definition": "movies with at least one positive training interaction",
            **audit["catalog"],
            **catalog_artifact,
        },
        "eligibility": audit["eligibility"],
        "negative_sampling": {
            "explicit_pool": ("catalog movies rated 1-2 by the user in train"),
            "unrated_pool": (
                "catalog movies with no training rating from the user, including ratings of 3"
            ),
            "fallback": (
                "users without eligible explicit negatives sample from the unrated pool only"
            ),
            **audit["negative_sampling"],
        },
        "experiments": _experiment_section(),
        "artifacts": {
            "splits": split_artifacts,
            "catalog": catalog_artifact,
        },
        "generated_by": "scripts/prepare_data.py",
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)

    print(f"wrote {args.output_dir}")
    print(f"wrote {args.manifest}")
    for split in SPLITS:
        artifact = split_artifacts[split]
        print(f"  {split}: {artifact['rows']:,} rows")
    print(f"  catalog: {catalog_artifact['rows']:,} movies")


if __name__ == "__main__":
    main()
