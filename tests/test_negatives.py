import numpy as np
import pandas as pd
import pytest
from src.data.negatives import NegativeSampler


def _frame(rows: list[tuple[int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "label"])


def _train() -> pd.DataFrame:
    return _frame(
        [
            (1, 1, "positive"),
            (1, 2, "positive"),
            (1, 3, "negative"),
            (1, 4, "neutral"),
            (2, 1, "positive"),
            (2, 2, "neutral"),
        ]
    )


def test_explicit_sampling_uses_only_the_explicit_pool() -> None:
    sampler = NegativeSampler(_train(), [1, 2, 3, 4, 5], 1.0, seed=7)

    sampled = sampler.sample()
    user_one = sampled.items[sampler.positive_user_ids == 1]
    user_two = sampled.items[sampler.positive_user_ids == 2]

    assert set(user_one.tolist()) == {3}
    assert set(user_two.tolist()) <= {3, 4, 5}
    assert sampled.explicit_count == 2
    assert sampled.realized_explicit_proportion == pytest.approx(2 / 3)


def test_unrated_sampling_excludes_every_training_rating() -> None:
    sampler = NegativeSampler(_train(), [1, 2, 3, 4, 5], 0.0, seed=7)

    sampled = sampler.sample()
    user_one = sampled.items[sampler.positive_user_ids == 1]
    user_two = sampled.items[sampler.positive_user_ids == 2]

    assert set(user_one.tolist()) == {5}
    assert set(user_two.tolist()) <= {3, 4, 5}


def test_same_seed_reproduces_negatives() -> None:
    first = NegativeSampler(_train(), [1, 2, 3, 4, 5], 0.5, seed=11).sample()
    second = NegativeSampler(_train(), [1, 2, 3, 4, 5], 0.5, seed=11).sample()

    assert np.array_equal(first.items, second.items)


def test_target_proportion_must_be_valid() -> None:
    with pytest.raises(ValueError, match="proportion"):
        NegativeSampler(_train(), [1, 2, 3], 1.5, seed=1)
