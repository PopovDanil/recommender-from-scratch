"""Dataset audits used by EDA and by the split manifest."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TypedDict

import pandas as pd

from src.data.splitting import (
    EXPLICIT_NEGATIVE,
    POSITIVE,
    TEST,
    TRAIN,
    VALIDATION,
    SplitBoundaries,
    build_catalog,
    prepare_ratings,
)
from src.experiments import EXPLICIT_NEGATIVE_PROPORTIONS


class CountSummary(TypedDict):
    min: float
    mean: float
    median: float
    max: float


class PeriodStats(TypedDict):
    n_ratings: int
    n_users: int
    n_movies: int
    n_positive: int
    n_negative: int
    n_neutral: int


class CatalogStats(TypedDict):
    n_movies: int
    train_positive_count: CountSummary


class EligibilityPeriod(TypedDict):
    positive_targets_total: int
    positive_targets_outside_catalog: int
    positive_targets_from_users_without_train_positive: int
    eligible_positive_targets: int
    target_users_in_catalog: int
    eligible_users: int
    excluded_users_without_train_positive: int
    excluded_users_without_eligible_target: int
    positives_per_eligible_user: CountSummary


class EligibilityStats(TypedDict):
    users_total: int
    train_users_with_positive: int
    users_without_train_positive: int
    periods: dict[str, EligibilityPeriod]


class ExplicitNegativeGroups(TypedDict):
    zero: int
    one_to_two: int
    three_plus: int


class NegativeSamplingStats(TypedDict):
    catalog_movies: int
    train_users_with_positive: int
    explicit_negatives_in_catalog: int
    users_with_explicit_negatives: int
    users_without_explicit_negatives: int
    explicit_negative_count_groups: ExplicitNegativeGroups
    explicit_pool_size: CountSummary
    unrated_pool_size: CountSummary
    expected_realized_proportion_if_uniform_over_triplets: dict[str, float]


class FullAudit(TypedDict):
    periods: dict[str, PeriodStats]
    catalog: CatalogStats
    eligibility: EligibilityStats
    negative_sampling: NegativeSamplingStats


@dataclass(frozen=True)
class CandidateMetrics:
    validation_start: datetime
    test_start: datetime
    train_share: float
    validation_share: float
    test_share: float
    catalog_movies: int
    train_users_with_positive: int
    users_without_train_positive: int
    validation_targets: int
    test_targets: int
    validation_targets_outside_catalog: int
    test_targets_outside_catalog: int
    validation_eligible_users: int
    test_eligible_users: int
    validation_eligible_targets: int
    test_eligible_targets: int
    explicit_negatives_in_catalog: int
    fallback_users: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_int_set(values: pd.Series) -> set[int]:
    return {int(value) for value in values.tolist()}


def _positive_targets(
    period: pd.DataFrame,
    catalog_ids: set[int],
    train_users: set[int],
) -> pd.DataFrame:
    return period[
        (period["label"] == POSITIVE)
        & period["movie_id"].isin(catalog_ids)
        & period["user_id"].isin(train_users)
    ]


def _describe_counts(counts: pd.Series) -> CountSummary:
    if counts.empty:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(counts.min()),
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "max": float(counts.max()),
    }


def candidate_metrics(
    ratings: pd.DataFrame,
    boundaries: SplitBoundaries,
) -> CandidateMetrics:
    """Summarize coverage for one pair of temporal boundaries."""
    prepared = prepare_ratings(ratings, boundaries)
    train = prepared[prepared["split"] == TRAIN]
    validation = prepared[prepared["split"] == VALIDATION]
    test = prepared[prepared["split"] == TEST]

    catalog = build_catalog(train)
    catalog_ids = _as_int_set(catalog["movie_id"])
    train_users = _as_int_set(train.loc[train["label"] == POSITIVE, "user_id"])
    total_users = int(prepared["user_id"].nunique())

    validation_eligible = _positive_targets(validation, catalog_ids, train_users)
    test_eligible = _positive_targets(test, catalog_ids, train_users)
    validation_positives = validation[validation["label"] == POSITIVE]
    test_positives = test[test["label"] == POSITIVE]

    explicit = train[(train["label"] == EXPLICIT_NEGATIVE) & train["movie_id"].isin(catalog_ids)]
    explicit_by_user = explicit.groupby("user_id").size()
    explicit_counts = {
        int(user_id): int(count)
        for user_id, count in zip(
            explicit_by_user.index.tolist(),
            explicit_by_user.tolist(),
            strict=True,
        )
    }
    fallback_users = sum(1 for user_id in train_users if explicit_counts.get(user_id, 0) == 0)

    total_ratings = len(prepared)
    return CandidateMetrics(
        validation_start=boundaries.validation_start,
        test_start=boundaries.test_start,
        train_share=len(train) / total_ratings,
        validation_share=len(validation) / total_ratings,
        test_share=len(test) / total_ratings,
        catalog_movies=len(catalog_ids),
        train_users_with_positive=len(train_users),
        users_without_train_positive=total_users - len(train_users),
        validation_targets=len(validation_positives),
        test_targets=len(test_positives),
        validation_targets_outside_catalog=len(validation_positives)
        - len(validation_positives[validation_positives["movie_id"].isin(catalog_ids)]),
        test_targets_outside_catalog=len(test_positives)
        - len(test_positives[test_positives["movie_id"].isin(catalog_ids)]),
        validation_eligible_users=int(validation_eligible["user_id"].nunique()),
        test_eligible_users=int(test_eligible["user_id"].nunique()),
        validation_eligible_targets=len(validation_eligible),
        test_eligible_targets=len(test_eligible),
        explicit_negatives_in_catalog=len(explicit),
        fallback_users=fallback_users,
    )


def period_summary(prepared: pd.DataFrame) -> dict[str, PeriodStats]:
    summary: dict[str, PeriodStats] = {}
    for split in (TRAIN, VALIDATION, TEST):
        period = prepared[prepared["split"] == split]
        summary[split] = {
            "n_ratings": len(period),
            "n_users": int(period["user_id"].nunique()),
            "n_movies": int(period["movie_id"].nunique()),
            "n_positive": int((period["label"] == POSITIVE).sum()),
            "n_negative": int((period["label"] == EXPLICIT_NEGATIVE).sum()),
            "n_neutral": int((period["label"] == "neutral").sum()),
        }
    return summary


def catalog_summary(catalog: pd.DataFrame) -> CatalogStats:
    return {
        "n_movies": len(catalog),
        "train_positive_count": _describe_counts(catalog["train_positive_count"]),
    }


def eligibility_summary(
    prepared: pd.DataFrame,
    catalog_ids: set[int],
) -> EligibilityStats:
    train = prepared[prepared["split"] == TRAIN]
    train_users = _as_int_set(train.loc[train["label"] == POSITIVE, "user_id"])
    total_users = int(prepared["user_id"].nunique())

    periods: dict[str, EligibilityPeriod] = {}
    for split in (VALIDATION, TEST):
        period = prepared[prepared["split"] == split]
        positives = period[period["label"] == POSITIVE]
        eligible = _positive_targets(period, catalog_ids, train_users)
        eligible_users = _as_int_set(eligible["user_id"])
        positives_in_catalog = positives[positives["movie_id"].isin(catalog_ids)]
        target_users = _as_int_set(positives_in_catalog["user_id"])
        users_without_history = _as_int_set(
            positives.loc[~positives["user_id"].isin(train_users), "user_id"]
        )

        periods[split] = {
            "positive_targets_total": len(positives),
            "positive_targets_outside_catalog": int(len(positives) - len(positives_in_catalog)),
            "positive_targets_from_users_without_train_positive": int(
                (~positives["user_id"].isin(train_users)).sum()
            ),
            "eligible_positive_targets": len(eligible),
            "target_users_in_catalog": len(target_users),
            "eligible_users": len(eligible_users),
            "excluded_users_without_train_positive": len(users_without_history),
            "excluded_users_without_eligible_target": len(train_users - eligible_users),
            "positives_per_eligible_user": _describe_counts(eligible.groupby("user_id").size()),
        }

    return {
        "users_total": total_users,
        "train_users_with_positive": len(train_users),
        "users_without_train_positive": total_users - len(train_users),
        "periods": periods,
    }


def negative_sampling_summary(
    prepared: pd.DataFrame,
    catalog_ids: set[int],
) -> NegativeSamplingStats:
    train = prepared[prepared["split"] == TRAIN]
    train_positives = train[train["label"] == POSITIVE]
    train_users = _as_int_set(train_positives["user_id"])

    rated_items: dict[int, set[int]] = {user_id: set() for user_id in train_users}
    explicit_counts: dict[int, int] = {user_id: 0 for user_id in train_users}
    positive_counts: dict[int, int] = {user_id: 0 for user_id in train_users}

    for user_id, movie_id, label in zip(
        train["user_id"].tolist(),
        train["movie_id"].tolist(),
        train["label"].tolist(),
        strict=True,
    ):
        user = int(user_id)
        movie = int(movie_id)
        if user not in rated_items or movie not in catalog_ids:
            continue
        rated_items[user].add(movie)
        if label == EXPLICIT_NEGATIVE:
            explicit_counts[user] += 1
        if label == POSITIVE:
            positive_counts[user] += 1

    explicit_sizes = pd.Series([explicit_counts[user] for user in train_users])
    unrated_sizes = pd.Series([len(catalog_ids) - len(rated_items[user]) for user in train_users])
    fallback_users = sum(1 for count in explicit_counts.values() if count == 0)
    total_positives = sum(positive_counts.values())
    positives_with_explicit = sum(
        count for user_id, count in positive_counts.items() if explicit_counts[user_id] > 0
    )

    expected_realized = {
        f"{proportion:g}": (
            proportion * positives_with_explicit / total_positives if total_positives else 0.0
        )
        for proportion in EXPLICIT_NEGATIVE_PROPORTIONS
    }

    return {
        "catalog_movies": len(catalog_ids),
        "train_users_with_positive": len(train_users),
        "explicit_negatives_in_catalog": sum(explicit_counts.values()),
        "users_with_explicit_negatives": len(train_users) - fallback_users,
        "users_without_explicit_negatives": fallback_users,
        "explicit_negative_count_groups": {
            "zero": fallback_users,
            "one_to_two": sum(1 for count in explicit_counts.values() if 1 <= count <= 2),
            "three_plus": sum(1 for count in explicit_counts.values() if count >= 3),
        },
        "explicit_pool_size": _describe_counts(explicit_sizes),
        "unrated_pool_size": _describe_counts(unrated_sizes),
        "expected_realized_proportion_if_uniform_over_triplets": expected_realized,
    }


def full_audit(
    prepared: pd.DataFrame,
    catalog: pd.DataFrame,
) -> FullAudit:
    catalog_ids = _as_int_set(catalog["movie_id"])
    return {
        "periods": period_summary(prepared),
        "catalog": catalog_summary(catalog),
        "eligibility": eligibility_summary(prepared, catalog_ids),
        "negative_sampling": negative_sampling_summary(prepared, catalog_ids),
    }
