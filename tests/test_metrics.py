import math
from collections.abc import Sequence

import pandas as pd
import pytest
from src.evaluation.metrics import (
    DEFAULT_METRICS,
    EvaluationSplit,
    Metric,
    evaluate,
    ndcg_at_k,
    recall_at_k,
)
from src.models.base import Recommender


class DictRecommender(Recommender):
    def __init__(self, scores: dict[int, dict[int, float]], default: float = 0.0) -> None:
        self._scores = scores
        self._default = default

    def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        user_scores = self._scores.get(user_id, {})
        return [user_scores.get(item, self._default) for item in item_ids]


class RecordingRecommender(Recommender):
    def __init__(self) -> None:
        self.requested: dict[int, list[int]] = {}

    def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        self.requested[user_id] = list(item_ids)
        return [float(item) for item in item_ids]


def _frame(rows: list[tuple[int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "label"])


def _toy_split() -> EvaluationSplit:
    train = _frame(
        [
            (1, 1, "positive"),
            (1, 3, "negative"),
            (1, 4, "neutral"),
            (2, 2, "positive"),
            (3, 1, "negative"),
        ]
    )
    validation = _frame(
        [
            (1, 2, "positive"),
            (1, 5, "positive"),
            (2, 5, "positive"),
            (3, 5, "positive"),
            (4, 1, "positive"),
        ]
    )
    return EvaluationSplit(
        train=train,
        validation=validation,
        test=validation.copy(),
        catalog=[1, 2, 3, 4, 5],
    )


def test_recall_at_k_counts_hits_over_relevant_items() -> None:
    ranked = [1, 2, 3, 4]

    assert recall_at_k(ranked, {1, 3, 5}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, {1, 3, 5}, 10) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, set(), 10) == 0.0


def test_ndcg_at_k_uses_ideal_ranking_for_normalization() -> None:
    ranked = [1, 2, 3, 4]
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    ideal = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)

    assert ndcg_at_k(ranked, {1, 3, 5}, 3) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(ranked, {1, 3, 5}, 10) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(ranked, set(), 10) == 0.0


def test_default_report_enables_every_metric() -> None:
    model = DictRecommender({1: {5: 2.0, 2: 1.0}, 2: {5: 1.0}})

    report = evaluate(_toy_split(), model)

    assert set(report) == {metric.name for metric in DEFAULT_METRICS}
    assert all(value == pytest.approx(1.0) for value in report.values())


def test_report_averages_metrics_equally_across_users() -> None:
    def hit_at_one(ranked: Sequence[int], relevant: set[int], k: int) -> float:
        return float(bool(ranked) and ranked[0] in relevant)

    model = DictRecommender(
        {
            1: {2: 1.0, 5: 2.0},
            2: {1: 4.0, 3: 3.0, 4: 2.0, 5: 1.0},
        }
    )
    extra = [Metric("hit@1", 1, hit_at_one)]

    report = evaluate(
        _toy_split(),
        model,
        metrics=["hit@1"],
        extra_metrics=extra,
    )

    assert report == {"hit@1": pytest.approx(0.5)}


def test_metrics_can_be_selected_and_excluded() -> None:
    model = DictRecommender({})

    only_recall = evaluate(_toy_split(), model, metrics=["recall@10"])
    without_ndcg_20 = evaluate(_toy_split(), model, exclude_metrics=["ndcg@20"])

    assert set(only_recall) == {"recall@10"}
    assert "ndcg@20" not in without_ndcg_20
    assert set(without_ndcg_20) == {"recall@10", "recall@20", "ndcg@10"}


def test_extra_metrics_join_the_default_set() -> None:
    def always_hit(ranked: Sequence[int], relevant: set[int], k: int) -> float:
        return 1.0

    report = evaluate(
        _toy_split(),
        DictRecommender({}),
        extra_metrics=[Metric("always", 1, always_hit)],
    )

    assert set(report) == {metric.name for metric in DEFAULT_METRICS} | {"always"}
    assert report["always"] == pytest.approx(1.0)


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate(_toy_split(), DictRecommender({}), metrics=["missing"])


def test_validation_masks_only_training_items() -> None:
    model = RecordingRecommender()

    evaluate(_toy_split(), model, period="validation", metrics=["recall@10"])

    assert model.requested[1] == [2, 5]
    assert model.requested[2] == [1, 3, 4, 5]


def test_test_masks_training_and_validation_items() -> None:
    train = _frame([(1, 1, "positive")])
    validation = _frame([(1, 2, "positive"), (1, 4, "negative")])
    test = _frame([(1, 5, "positive")])
    split = EvaluationSplit(
        train=train,
        validation=validation,
        test=test,
        catalog=[1, 2, 3, 4, 5],
    )
    model = RecordingRecommender()

    evaluate(split, model, period="test", metrics=["recall@10"])

    assert model.requested[1] == [3, 5]


def test_users_without_training_positive_are_excluded() -> None:
    model = RecordingRecommender()

    evaluate(_toy_split(), model, metrics=["recall@10"])

    assert set(model.requested) == {1, 2}


def test_model_must_return_one_score_per_item() -> None:
    class ShortRecommender(Recommender):
        def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
            return []

    with pytest.raises(ValueError, match="scores"):
        evaluate(_toy_split(), ShortRecommender())


def test_model_must_return_finite_scores() -> None:
    class InfiniteRecommender(Recommender):
        def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
            return [math.inf for _ in item_ids]

    with pytest.raises(ValueError, match="non-finite"):
        evaluate(_toy_split(), InfiniteRecommender())


def test_no_eligible_users_yields_zero_metrics() -> None:
    empty = _frame([])
    split = EvaluationSplit(train=empty, validation=empty, test=empty, catalog=[1, 2])

    report = evaluate(split, DictRecommender({}))

    assert set(report) == {metric.name for metric in DEFAULT_METRICS}
    assert all(value == 0.0 for value in report.values())
