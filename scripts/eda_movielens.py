"""Generate the MovieLens-1M data audit report.

Run from the repository root:

    uv run python -m scripts.eda_movielens

The report documents the coverage audit used to freeze the temporal split. It
reads only the raw release and writes ``docs/data-audit.md``; it does not create
model training artifacts.
"""

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from src.data.audit import (
    CandidateMetrics,
    candidate_metrics,
    catalog_summary,
    eligibility_summary,
    negative_sampling_summary,
    period_summary,
)
from src.data.loading import load_movies, load_ratings, load_users
from src.data.splitting import (
    DEFAULT_BOUNDARIES,
    SPLITS,
    SplitBoundaries,
    build_catalog,
    prepare_ratings,
)

ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_PAIRS = (
    ("2000-08-01", "2000-11-01"),
    ("2000-09-01", "2000-12-01"),
    ("2000-09-01", "2001-01-01"),
    ("2000-10-01", "2000-12-01"),
    ("2000-10-01", "2001-01-01"),
    ("2000-10-01", "2001-04-01"),
    ("2000-11-01", "2001-04-01"),
    ("2000-12-01", "2001-04-01"),
    ("2000-12-01", "2001-07-01"),
    ("2001-01-01", "2001-07-01"),
    ("2001-04-01", "2001-10-01"),
    ("2001-07-01", "2002-01-01"),
    ("2002-01-01", "2002-07-01"),
)


def _boundaries(validation_start: str, test_start: str) -> SplitBoundaries:
    return SplitBoundaries(
        validation_start=datetime.fromisoformat(validation_start).replace(tzinfo=UTC),
        test_start=datetime.fromisoformat(test_start).replace(tzinfo=UTC),
    )


def _date(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%d")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _quantile_rows(series: pd.Series) -> list[str]:
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    return [f"{series.quantile(q):.0f}" for q in quantiles] + [f"{series.mean():.1f}"]


def _candidate_tables(metrics: Sequence[CandidateMetrics]) -> tuple[str, str]:
    share_rows = [
        [
            m.validation_start.strftime("%Y-%m-%d"),
            m.test_start.strftime("%Y-%m-%d"),
            f"{m.train_share:.1%}",
            f"{m.validation_share:.1%}",
            f"{m.test_share:.1%}",
            f"{m.catalog_movies:,}",
            f"{m.train_users_with_positive:,}",
            f"{m.users_without_train_positive:,}",
        ]
        for m in metrics
    ]
    coverage_rows = [
        [
            m.validation_start.strftime("%Y-%m-%d"),
            m.test_start.strftime("%Y-%m-%d"),
            f"{m.validation_targets:,}",
            f"{m.test_targets:,}",
            f"{m.validation_eligible_users:,}",
            f"{m.test_eligible_users:,}",
            f"{m.validation_eligible_targets:,}",
            f"{m.test_eligible_targets:,}",
            f"{m.validation_targets_outside_catalog:,}",
            f"{m.test_targets_outside_catalog:,}",
            f"{m.explicit_negatives_in_catalog:,}",
            f"{m.fallback_users:,}",
        ]
        for m in metrics
    ]
    first = _table(
        (
            "val start",
            "test start",
            "train",
            "val",
            "test",
            "catalog",
            "train users",
            "no train pos",
        ),
        share_rows,
    )
    second = _table(
        (
            "val start",
            "test start",
            "val targets",
            "test targets",
            "val users",
            "test users",
            "val eligible",
            "test eligible",
            "val outside",
            "test outside",
            "explicit neg",
            "fallback",
        ),
        coverage_rows,
    )
    return first, second


def _render_report(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    metrics: Sequence[CandidateMetrics],
) -> str:
    prepared = prepare_ratings(ratings)
    train = prepared[prepared["split"] == "train"]
    catalog = build_catalog(train)
    catalog_ids = {int(movie_id) for movie_id in catalog["movie_id"].tolist()}

    periods = period_summary(prepared)
    catalog_stats = catalog_summary(catalog)
    eligibility = eligibility_summary(prepared, catalog_ids)
    negative = negative_sampling_summary(prepared, catalog_ids)
    selected = candidate_metrics(ratings, DEFAULT_BOUNDARIES)

    n_users = int(ratings["user_id"].nunique())
    n_rated_movies = int(ratings["movie_id"].nunique())
    sparsity = 1.0 - len(ratings) / (n_users * n_rated_movies)
    timestamp_min = _date(int(ratings["timestamp"].min()))
    timestamp_max = _date(int(ratings["timestamp"].max()))

    rating_rows = [
        [
            str(rating),
            f"{int(count):,}",
            f"{count / len(ratings):.2%}",
        ]
        for rating, count in ratings["rating"].value_counts().sort_index().items()
    ]
    year_rows = [
        [str(year), f"{int(count):,}", f"{count / len(ratings):.2%}"]
        for year, count in ratings["timestamp"]
        .map(lambda value: datetime.fromtimestamp(value, tz=UTC).year)
        .value_counts()
        .sort_index()
        .items()
    ]

    ratings_per_user = ratings.groupby("user_id").size()
    positives_per_user = (
        ratings[ratings["rating"] >= 4]
        .groupby("user_id")
        .size()
        .reindex(ratings["user_id"].unique(), fill_value=0)
    )
    quantile_headers = ("min", "p25", "p50", "p75", "p90", "p99", "max", "mean")

    first_candidates, second_candidates = _candidate_tables(metrics)

    period_rows = [
        [
            split,
            f"{periods[split]['n_ratings']:,}",
            f"{periods[split]['n_users']:,}",
            f"{periods[split]['n_movies']:,}",
            f"{periods[split]['n_positive']:,}",
            f"{periods[split]['n_negative']:,}",
            f"{periods[split]['n_neutral']:,}",
        ]
        for split in SPLITS
    ]

    eligibility_rows = []
    for split in ("validation", "test"):
        period = eligibility["periods"][split]
        per_user = period["positives_per_eligible_user"]
        eligibility_rows.append(
            [
                split,
                f"{period['positive_targets_total']:,}",
                f"{period['positive_targets_outside_catalog']:,}",
                f"{period['positive_targets_from_users_without_train_positive']:,}",
                f"{period['eligible_positive_targets']:,}",
                f"{period['eligible_users']:,}",
                f"{period['excluded_users_without_train_positive']:,}",
                f"{period['excluded_users_without_eligible_target']:,}",
                f"{per_user['mean']:.1f} / {per_user['median']:.0f} / "
                f"{per_user['min']:.0f}-{per_user['max']:.0f}",
            ]
        )

    explicit_groups = negative["explicit_negative_count_groups"]
    expected = negative["expected_realized_proportion_if_uniform_over_triplets"]

    period_table = _table(
        ("split", "ratings", "users", "movies", "positives", "explicit neg", "neutral"),
        period_rows,
    )
    eligibility_table = _table(
        (
            "split",
            "positives",
            "outside catalog",
            "no train history",
            "eligible targets",
            "eligible users",
            "excluded no history",
            "excluded no target",
            "pos/user mean / median / range",
        ),
        eligibility_rows,
    )
    explicit_group_text = (
        f"{explicit_groups['zero']:,} / {explicit_groups['one_to_two']:,} / "
        f"{explicit_groups['three_plus']:,}"
    )
    explicit_pool_text = (
        f"{negative['explicit_pool_size']['min']:.0f} / "
        f"{negative['explicit_pool_size']['median']:.0f} / "
        f"{negative['explicit_pool_size']['mean']:.1f} / "
        f"{negative['explicit_pool_size']['max']:.0f}"
    )
    unrated_pool_text = (
        f"{negative['unrated_pool_size']['min']:.0f} / "
        f"{negative['unrated_pool_size']['median']:.0f} / "
        f"{negative['unrated_pool_size']['mean']:.1f} / "
        f"{negative['unrated_pool_size']['max']:.0f}"
    )

    return f"""# MovieLens-1M data audit

Generated by `scripts/eda_movielens.py` from `data/ml-1m/`. Raw data and generated
splits are not committed.

## Dataset overview

| item | value |
| --- | --- |
| ratings | {len(ratings):,} |
| users | {n_users:,} |
| rated movies | {n_rated_movies:,} |
| movies in release | {len(movies):,} |
| users file rows | {len(users):,} |
| timestamp range | {timestamp_min} to {timestamp_max} |
| user-movie sparsity | {sparsity:.4%} |

### Rating distribution

{_table(("rating", "count", "share"), rating_rows)}

### Ratings by year

{_table(("year", "count", "share"), year_rows)}

### Per-user distributions

| quantity | {" | ".join(quantile_headers)} |
| --- | {" | ".join("---" for _ in quantile_headers)} |
| ratings per user | {" | ".join(_quantile_rows(ratings_per_user))} |
| positive ratings per user | {" | ".join(_quantile_rows(positives_per_user))} |

The release is heavily concentrated in 2000: {year_rows[0][2]} of ratings are from
2000, so boundaries based on equal calendar intervals would produce highly unbalanced
periods. The audit therefore compares observed-timestamp boundaries directly.

## Boundary candidates

Training period shares and catalog size:

{first_candidates}

Held-out coverage. `val/test targets` count all positives in the period; `eligible`
counts the positives that remain after requiring a training positive for the user and
membership in the training catalog. `outside` counts positive targets excluded because
they are not in the catalog.

{second_candidates}

The `explicit neg` column counts all training ratings of 1-2 on catalog movies. The
negative-sampling section restricts the count to users with a training positive, because
only those users can contribute training triplets.

### Selected boundaries

- Validation starts at **{DEFAULT_BOUNDARIES.validation_start.isoformat()}**.
- Test starts at **{DEFAULT_BOUNDARIES.test_start.isoformat()}**.
- Assignment: `timestamp < validation_start` is train, timestamps in
  `[validation_start, test_start)` are validation, and timestamps at or after
  `test_start` are test. Equal timestamps therefore stay in the same split.

The selected pair keeps {selected.train_users_with_positive:,} of {n_users:,} users with a
training positive, while retaining {selected.validation_eligible_users:,} validation and
{selected.test_eligible_users:,} test eligible users. Earlier boundaries increase held-out
volume but leave hundreds of users without training positives, which would remove them
from the primary evaluation population. Later boundaries shrink validation and test too
much. The boundaries are frozen before model tuning.

## Selected split

{period_table}

### Catalog

The catalog is every movie with at least one positive training interaction.

| item | value |
| --- | --- |
| catalog movies | {catalog_stats["n_movies"]:,} |
| train positives per movie (min) | {catalog_stats["train_positive_count"]["min"]:.0f} |
| train positives per movie (median) | {catalog_stats["train_positive_count"]["median"]:.0f} |
| train positives per movie (mean) | {catalog_stats["train_positive_count"]["mean"]:.1f} |
| train positives per movie (max) | {catalog_stats["train_positive_count"]["max"]:.0f} |

### Evaluation eligibility

{eligibility_table}

Users without a training positive are excluded from the primary metrics. Users with a
training positive but no eligible held-out target are also excluded per period.

### Negative sampling pools

| item | value |
| --- | --- |
| explicit negatives in catalog | {negative["explicit_negatives_in_catalog"]:,} |
| users with explicit negatives | {negative["users_with_explicit_negatives"]:,} |
| users without explicit negatives | {negative["users_without_explicit_negatives"]:,} |
| users with zero / 1-2 / 3+ explicit negatives | {explicit_group_text} |
| explicit pool size (min / median / mean / max) | {explicit_pool_text} |
| unrated pool size (min / median / mean / max) | {unrated_pool_text} |

Expected realized explicit-negative proportion if each positive contributes one triplet:

| target | expected realized |
| --- | --- |
| 0% | {expected["0"]:.2%} |
| 25% | {expected["0.25"]:.2%} |
| 50% | {expected["0.5"]:.2%} |

The realized proportion can be lower than the target because users without eligible
explicit negatives fall back to the unrated pool. Training runs must log the realized
value rather than assume the target.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "ml-1m",
        help="directory containing ratings.dat, users.dat and movies.dat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "data-audit.md",
        help="markdown report path",
    )
    args = parser.parse_args()

    ratings = load_ratings(args.data_dir / "ratings.dat")
    users = load_users(args.data_dir / "users.dat")
    movies = load_movies(args.data_dir / "movies.dat")

    metrics = [
        candidate_metrics(ratings, _boundaries(validation, test))
        for validation, test in CANDIDATE_PAIRS
    ]
    report = _render_report(ratings, users, movies, metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    selected = candidate_metrics(ratings, DEFAULT_BOUNDARIES)
    print(f"wrote {args.output}")
    print(
        f"selected {DEFAULT_BOUNDARIES.validation_start.date()} / "
        f"{DEFAULT_BOUNDARIES.test_start.date()}: "
        f"catalog={selected.catalog_movies:,}, "
        f"validation users={selected.validation_eligible_users:,}, "
        f"test users={selected.test_eligible_users:,}, "
        f"fallback users={selected.fallback_users:,}"
    )


if __name__ == "__main__":
    main()
