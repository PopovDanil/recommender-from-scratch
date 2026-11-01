"""Most-popular baseline ranked by positive training interactions."""

from collections.abc import Mapping, Sequence

import pandas as pd

from src.data.splitting import POSITIVE
from src.models.base import Recommender


class Popularity(Recommender):
    """Score every catalog item by its number of positive training ratings."""

    def __init__(self, positive_counts: Mapping[int, float]) -> None:
        self._positive_counts = {int(item): float(count) for item, count in positive_counts.items()}

    @classmethod
    def from_train(
        cls,
        train_ratings: pd.DataFrame,
        catalog_ids: Sequence[int],
    ) -> "Popularity":
        positives = train_ratings[train_ratings["label"] == POSITIVE]
        counts = positives.groupby("movie_id").size()
        return cls({int(item): float(counts.get(int(item), 0.0)) for item in catalog_ids})

    def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        return [self._positive_counts.get(int(item), 0.0) for item in item_ids]
