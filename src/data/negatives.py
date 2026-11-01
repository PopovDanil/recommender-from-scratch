"""Training negative sampler for the BPR experiments.

For every positive training interaction the sampler draws one negative from
either the user's explicit-negative pool (catalog movies rated 1-2 in train) or
the unrated pool (catalog movies with no training rating from the user). Users
without eligible explicit negatives always fall back to the unrated pool.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.data.splitting import EXPLICIT_NEGATIVE, POSITIVE

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SampledNegatives:
    """Negative item ids aligned with the sampler's positive interactions."""

    items: IntArray
    explicit_count: int

    @property
    def realized_explicit_proportion(self) -> float:
        if self.items.size == 0:
            return 0.0
        return self.explicit_count / self.items.size


class NegativeSampler:
    """Draw BPR negatives from the training-only explicit and unrated pools."""

    def __init__(
        self,
        train_ratings: pd.DataFrame,
        catalog_ids: Sequence[int],
        explicit_negative_proportion: float,
        *,
        seed: int,
    ) -> None:
        if not 0.0 <= explicit_negative_proportion <= 1.0:
            raise ValueError("explicit negative proportion must be in [0, 1]")

        self._proportion = float(explicit_negative_proportion)
        self._rng = np.random.default_rng(seed)
        self._catalog = np.array(sorted(int(item) for item in catalog_ids), dtype=np.int64)
        if self._catalog.size == 0:
            raise ValueError("catalog must contain at least one item")

        catalog_index = {int(item): index for index, item in enumerate(self._catalog)}
        catalog_ratings = train_ratings[train_ratings["movie_id"].isin(catalog_index)]
        positives = catalog_ratings[catalog_ratings["label"] == POSITIVE]
        ratings_by_user = {
            int(cast(int, user_id)): group
            for user_id, group in catalog_ratings.groupby("user_id", sort=False)
        }

        users: list[int] = []
        positives_by_user: list[IntArray] = []
        explicit_by_user: list[IntArray] = []
        rated_by_user: list[set[int]] = []

        for user_id, group in positives.groupby("user_id", sort=True):
            user = int(cast(int, user_id))
            all_ratings = ratings_by_user[user]
            rated_items = {catalog_index[int(item)] for item in all_ratings["movie_id"]}
            explicit_items = _catalog_positions(
                all_ratings.loc[all_ratings["label"] == EXPLICIT_NEGATIVE, "movie_id"].tolist(),
                catalog_index,
            )

            users.append(user)
            positives_by_user.append(_catalog_positions(group["movie_id"].tolist(), catalog_index))
            explicit_by_user.append(explicit_items)
            rated_by_user.append(rated_items)

        self._users = np.array(users, dtype=np.int64)
        self._sizes = np.array([items.size for items in positives_by_user], dtype=np.int64)
        self._offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(self._sizes)))
        self._positive_items = (
            np.concatenate(positives_by_user).astype(np.int64, copy=False)
            if positives_by_user
            else np.empty(0, dtype=np.int64)
        )
        self._explicit = explicit_by_user
        self._rated = rated_by_user
        self._positive_users = np.repeat(self._users, self._sizes)

    @property
    def positive_user_ids(self) -> IntArray:
        return self._positive_users

    @property
    def positive_item_ids(self) -> IntArray:
        return np.array(self._catalog[self._positive_items], dtype=np.int64)

    @property
    def n_positives(self) -> int:
        return int(self._positive_users.size)

    def sample(self) -> SampledNegatives:
        """Draw one negative per positive interaction for a training epoch."""
        negatives = np.empty(self.n_positives, dtype=np.int64)
        explicit_count = 0

        for group_index, _user_id in enumerate(self._users):
            start = int(self._offsets[group_index])
            stop = int(self._offsets[group_index + 1])
            block = negatives[start:stop]
            explicit = self._explicit[group_index]

            if explicit.size > 0 and self._proportion > 0.0:
                use_explicit = self._rng.random(block.size) < self._proportion
                n_explicit = int(use_explicit.sum())
                if n_explicit > 0:
                    block[use_explicit] = self._catalog[
                        explicit[self._rng.integers(0, explicit.size, size=n_explicit)]
                    ]
                    explicit_count += n_explicit
                n_unrated = block.size - n_explicit
            else:
                use_explicit = np.zeros(block.size, dtype=bool)
                n_unrated = block.size

            if n_unrated > 0:
                block[~use_explicit] = self._sample_unrated(group_index, n_unrated)

        return SampledNegatives(items=negatives, explicit_count=explicit_count)

    def _sample_unrated(self, group_index: int, size: int) -> IntArray:
        rated = self._rated[group_index]
        if len(rated) >= self._catalog.size:
            raise ValueError("user has no unrated catalog items to sample")

        positions = self._rng.integers(0, self._catalog.size, size=size)
        rejected = np.array([int(position) in rated for position in positions], dtype=bool)
        while rejected.any():
            n_rejected = int(rejected.sum())
            positions[rejected] = self._rng.integers(0, self._catalog.size, size=n_rejected)
            rejected = np.array([int(position) in rated for position in positions], dtype=bool)
        return self._catalog[positions]


def _catalog_positions(items: Sequence[int], catalog_index: dict[int, int]) -> IntArray:
    """Map external movie ids to catalog positions, keeping input order."""
    return np.array([catalog_index[int(item)] for item in items], dtype=np.int64)
