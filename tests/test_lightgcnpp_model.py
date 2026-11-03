import numpy as np
import pytest
import torch
from src.models.base import Recommender
from src.models.lightgcnpp import LightGCNPlusPlus


def _model(
    *,
    gamma: float = 0.2,
    n_layers: int = 1,
    scale: float = 1.0,
) -> LightGCNPlusPlus:
    model = LightGCNPlusPlus(
        [10, 20],
        [1, 2],
        np.array([10, 10, 20, 20]),
        np.array([1, 2, 1, 2]),
        n_layers=n_layers,
        embedding_dim=2,
        init_std=0.0,
        alpha=0.6,
        beta=-0.1,
        gamma=gamma,
    )
    with torch.no_grad():
        model.embedding.weight.copy_(
            scale
            * torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )
    return model


def test_lightgcnpp_satisfies_recommender_interface() -> None:
    assert isinstance(_model(), Recommender)


def test_gamma_one_scores_with_the_input_layer() -> None:
    scores = _model(gamma=1.0).score(10, [1, 2, 99])

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == 0.0


def test_unit_normalization_makes_propagated_scores_scale_invariant() -> None:
    baseline = _model(gamma=0.0).score(10, [1, 2])
    scaled = _model(gamma=0.0, scale=10.0).score(10, [1, 2])

    assert scaled == pytest.approx(baseline)


def test_unknown_users_score_zero() -> None:
    assert list(_model().score(999, [1, 2])) == [0.0, 0.0]


def test_gamma_must_be_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="gamma"):
        _model(gamma=1.5)
