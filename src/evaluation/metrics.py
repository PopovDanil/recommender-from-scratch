"""Full-catalog ranking metrics and the shared evaluator.

Metrics are computed from one ranking per eligible user and averaged equally
across users. ``top_k`` is inclusive and lower rank is better. A user is
eligible when they have at least one positive training interaction and at least
one positive target in the evaluated period that remains in the catalog after
user-specific masking.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.data.splitting import POSITIVE, TEST, VALIDATION
from src.models.base import Recommender

MetricFunction = Callable[[Sequence[int], set[int], int], float]


@dataclass(frozen=True)
class Metric:
    """One ranking metric evaluated at a fixed cutoff."""

    name: str
    k: int
    function: MetricFunction


@dataclass(frozen=True)
class EvaluationSplit:
    """Materialized temporal split and the shared catalog used for evaluation."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    catalog: pd.DataFrame | Sequence[int]


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of relevant items that appear in the first ``k`` positions."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG at ``k`` normalized by the ideal ranking."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


DEFAULT_METRICS: tuple[Metric, ...] = (
    Metric("recall@10", 10, recall_at_k),
    Metric("recall@20", 20, recall_at_k),
    Metric("ndcg@10", 10, ndcg_at_k),
    Metric("ndcg@20", 20, ndcg_at_k),
)


def _catalog_ids(catalog: pd.DataFrame | Sequence[int]) -> tuple[int, ...]:
    if isinstance(catalog, pd.DataFrame):
        return tuple(sorted(int(item) for item in catalog["movie_id"].tolist()))
    return tuple(sorted(int(item) for item in catalog))


def _resolve_metrics(
    metrics: Iterable[str] | None,
    exclude_metrics: Iterable[str],
    extra_metrics: Sequence[Metric] | None,
) -> list[Metric]:
    registry = {metric.name: metric for metric in DEFAULT_METRICS}
    for metric in extra_metrics or ():
        registry[metric.name] = metric

    selected = list(registry) if metrics is None else list(metrics)
    excluded = set(exclude_metrics)
    unknown = (set(selected) | excluded) - set(registry)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown metric(s): {names}")

    return [registry[name] for name in selected if name not in excluded]


def _period_frame(split: EvaluationSplit, period: str) -> pd.DataFrame:
    if period == VALIDATION:
        return split.validation
    if period == TEST:
        return split.test
    raise ValueError(f"period must be {VALIDATION!r} or {TEST!r}, got {period!r}")


def _history_frame(split: EvaluationSplit, period: str) -> pd.DataFrame:
    if period == VALIDATION:
        return split.train
    return pd.concat([split.train, split.validation], ignore_index=True)


def _masked_items(history: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(cast(int, user_id)): {int(item) for item in items}
        for user_id, items in history.groupby("user_id")["movie_id"]
    }


def _eligible_targets(
    period: pd.DataFrame,
    catalog_ids: set[int],
    train_users: set[int],
) -> dict[int, set[int]]:
    positives = period[
        (period["label"] == POSITIVE)
        & period["movie_id"].isin(catalog_ids)
        & period["user_id"].isin(train_users)
    ]
    return {
        int(cast(int, user_id)): {int(item) for item in items}
        for user_id, items in positives.groupby("user_id")["movie_id"]
    }


def evaluate(
    split: EvaluationSplit,
    model: Recommender,
    *,
    period: str = VALIDATION,
    metrics: Iterable[str] | None = None,
    exclude_metrics: Iterable[str] = (),
    extra_metrics: Sequence[Metric] | None = None,
) -> dict[str, float]:
    """Evaluate ``model`` on one held-out period and average over eligible users.

    All metrics are enabled by default. Pass ``metrics`` to evaluate a subset,
    ``exclude_metrics`` to drop names from the selected set, or ``extra_metrics``
    to register additional metrics. At validation the candidate set masks every
    movie rated by the user in train; at test it masks every movie rated in train
    or validation. The ranking always covers the full remaining catalog.
    """
    selected_metrics = _resolve_metrics(metrics, exclude_metrics, extra_metrics)
    period_frame = _period_frame(split, period)
    catalog = _catalog_ids(split.catalog)
    catalog_ids = set(catalog)

    train_users = {
        int(user_id)
        for user_id in split.train.loc[split.train["label"] == POSITIVE, "user_id"].tolist()
    }
    masked_by_user = _masked_items(_history_frame(split, period))
    targets_by_user = _eligible_targets(period_frame, catalog_ids, train_users)

    totals = {metric.name: 0.0 for metric in selected_metrics}
    eligible_users = 0

    for user_id, target_items in targets_by_user.items():
        masked = masked_by_user.get(user_id, set())
        candidate_ids = [item for item in catalog if item not in masked]
        relevant = target_items & set(candidate_ids)
        if not relevant:
            continue

        scores = model.score(user_id, candidate_ids)
        if len(scores) != len(candidate_ids):
            raise ValueError(
                f"model returned {len(scores)} scores for user {user_id}, "
                f"expected {len(candidate_ids)}"
            )
        if not all(math.isfinite(float(score)) for score in scores):
            raise ValueError(f"model returned a non-finite score for user {user_id}")

        ranked = [
            item
            for _, item in sorted(
                zip(scores, candidate_ids, strict=True),
                key=lambda pair: (-pair[0], pair[1]),
            )
        ]
        for metric in selected_metrics:
            totals[metric.name] += metric.function(ranked, relevant, metric.k)
        eligible_users += 1

    if eligible_users == 0:
        return {metric.name: 0.0 for metric in selected_metrics}

    return {name: total / eligible_users for name, total in totals.items()}


evaluate_split = evaluate
