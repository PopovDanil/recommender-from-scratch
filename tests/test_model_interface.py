from collections.abc import Sequence

import pytest
from src.models.base import Recommender


class StaticRecommender(Recommender):
    def __init__(self, scores: dict[int, float]) -> None:
        self._scores = scores

    def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        return [self._scores[item] for item in item_ids]


def test_recommender_is_abstract() -> None:
    with pytest.raises(TypeError):
        Recommender()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated() -> None:
    class Incomplete(Recommender):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_concrete_model_scores_items_in_requested_order() -> None:
    model = StaticRecommender({1: 0.5, 2: 1.5})

    assert list(model.score(7, [2, 1])) == [1.5, 0.5]
