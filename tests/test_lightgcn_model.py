import math

import numpy as np
import pandas as pd
import pytest
import torch
from src.models.base import Recommender
from src.models.lightgcn import LightGCN, positive_interaction_edges


def _model(n_layers: int = 1) -> LightGCN:
    model = LightGCN(
        [10, 20],
        [1, 2],
        np.array([10, 10, 20, 20]),
        np.array([1, 2, 1, 2]),
        n_layers=n_layers,
        embedding_dim=2,
        init_std=0.0,
    )
    with torch.no_grad():
        model.embedding.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )
    return model


def test_lightgcn_satisfies_recommender_interface() -> None:
    assert isinstance(_model(), Recommender)


def test_score_averages_input_and_propagated_layers() -> None:
    model = _model(n_layers=1)

    scores = model.score(10, [1, 2, 99])

    assert scores[0] == pytest.approx(0.625)
    assert scores[1] == pytest.approx(0.375)
    assert scores[0] > scores[1]
    assert scores[2] == 0.0
    assert all(math.isfinite(score) for score in scores)


def test_unknown_users_score_zero() -> None:
    assert list(_model().score(999, [1, 2])) == [0.0, 0.0]


def test_encode_items_offsets_into_the_node_matrix() -> None:
    model = _model()

    assert model.encode_users(np.array([20, 10])).tolist() == [1, 0]
    assert model.encode_items(np.array([2, 1])).tolist() == [3, 2]


def test_bpr_loss_rewards_positive_above_negative() -> None:
    model = _model()
    users = torch.tensor([0])
    positives = torch.tensor([2])
    negatives = torch.tensor([3])

    good = model.bpr_loss(users, positives, negatives)
    bad = model.bpr_loss(users, negatives, positives)

    assert good.item() < bad.item()


def test_positive_interaction_edges_keep_only_catalog_positives() -> None:
    train = pd.DataFrame(
        [
            (10, 1, "positive"),
            (10, 2, "negative"),
            (20, 2, "positive"),
            (20, 3, "positive"),
            (20, 4, "neutral"),
        ],
        columns=["user_id", "movie_id", "label"],
    )

    users, items = positive_interaction_edges(train, [1, 2, 3])

    assert users.tolist() == [10, 20, 20]
    assert items.tolist() == [1, 2, 3]


def test_graph_requires_at_least_one_edge() -> None:
    with pytest.raises(ValueError, match="positive edge"):
        LightGCN([10], [1], np.array([], dtype=np.int64), np.array([], dtype=np.int64))
