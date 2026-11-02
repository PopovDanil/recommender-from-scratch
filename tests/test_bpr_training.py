import math

import numpy as np
import pandas as pd
from src.data.negatives import NegativeSampler
from src.models.bpr_mf import BPRMatrixFactorization
from src.models.lightgcn import LightGCN
from src.training.bpr import BPRTrainingConfig, train_bpr


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, 1, "positive"),
            (1, 2, "positive"),
            (1, 3, "negative"),
            (2, 1, "positive"),
            (2, 2, "positive"),
        ],
        columns=["user_id", "movie_id", "label"],
    )


def test_training_runs_one_triplet_per_positive_per_epoch() -> None:
    catalog = [1, 2, 3, 4]
    sampler = NegativeSampler(_train(), catalog, 1.0, seed=3)
    model = BPRMatrixFactorization([1, 2], catalog, embedding_dim=4)
    config = BPRTrainingConfig(
        epochs=2,
        batch_size=2,
        learning_rate=1e-2,
        regularization=1e-5,
    )

    history = train_bpr(model, sampler, config, seed=3)

    assert len(history) == 2
    assert all(math.isfinite(report.loss) for report in history)
    assert all(report.realized_explicit_proportion == 0.5 for report in history)
    assert all(report.n_updates == 2 for report in history)


def test_callback_can_stop_training_early() -> None:
    catalog = [1, 2, 3, 4]
    sampler = NegativeSampler(_train(), catalog, 0.0, seed=3)
    model = BPRMatrixFactorization([1, 2], catalog, embedding_dim=4)
    config = BPRTrainingConfig(epochs=10, batch_size=2)

    history = train_bpr(
        model,
        sampler,
        config,
        seed=3,
        on_epoch_end=lambda report: report.epoch == 1,
    )

    assert len(history) == 1


def test_lightgcn_training_updates_embeddings() -> None:
    catalog = [1, 2, 3, 4]
    sampler = NegativeSampler(_train(), catalog, 0.0, seed=3)
    model = LightGCN(
        [1, 2],
        catalog,
        np.array([1, 1, 2, 2]),
        np.array([1, 2, 1, 2]),
        n_layers=2,
        embedding_dim=4,
    )
    before = model.embedding.weight.detach().clone()
    config = BPRTrainingConfig(epochs=2, batch_size=2, learning_rate=1e-2)

    history = train_bpr(model, sampler, config, seed=3)

    assert len(history) == 2
    assert all(math.isfinite(report.loss) for report in history)
    assert not np.allclose(before.numpy(), model.embedding.weight.detach().numpy())
