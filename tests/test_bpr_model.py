import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from src.models.base import Recommender
from src.models.bpr_mf import BPRMatrixFactorization
from src.models.popularity import Popularity


def _model() -> BPRMatrixFactorization:
    model = BPRMatrixFactorization([10, 20], [1, 2, 3], embedding_dim=2, init_std=0.0)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.item_embedding.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    return model


def test_bpr_model_satisfies_recommender_interface() -> None:
    assert isinstance(_model(), Recommender)


def test_score_uses_dot_product_and_handles_unknown_items() -> None:
    model = _model()

    scores = model.score(10, [2, 1, 99])

    assert list(scores) == [0.0, 1.0, 0.0]
    assert all(math.isfinite(score) for score in scores)


def test_unknown_users_score_zero() -> None:
    assert list(_model().score(999, [1, 2])) == [0.0, 0.0]


def test_encode_maps_external_ids_to_embedding_rows() -> None:
    model = _model()

    assert model.encode_users(np.array([20, 10])).tolist() == [1, 0]
    assert model.encode_items(np.array([3, 1])).tolist() == [2, 0]


def test_bpr_loss_rewards_positive_above_negative() -> None:
    model = _model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([1])

    good = model.bpr_loss(users, positives, negatives)
    bad = model.bpr_loss(users, negatives, positives)

    assert good.item() < bad.item()


def test_popularity_counts_positive_training_interactions() -> None:
    train = pd.DataFrame(
        [
            (1, 5, "positive"),
            (2, 5, "positive"),
            (1, 6, "positive"),
            (1, 7, "negative"),
        ],
        columns=["user_id", "movie_id", "label"],
    )
    model = Popularity.from_train(train, [5, 6, 7])

    assert list(model.score(1, [5, 6, 7])) == [2.0, 1.0, 0.0]


def test_popularity_scores_in_requested_order() -> None:
    def scored(model: Recommender) -> Sequence[float]:
        return model.score(1, [7, 5])

    assert list(scored(Popularity({5: 3.0, 7: 1.0}))) == [1.0, 3.0]
