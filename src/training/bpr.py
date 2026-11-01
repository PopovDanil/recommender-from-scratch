"""Mini-batch BPR training with one triplet per positive interaction per epoch."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from src.data.negatives import NegativeSampler
from src.models.bpr_mf import BPRMatrixFactorization


@dataclass(frozen=True)
class BPRTrainingConfig:
    """Optimization settings shared by every BPR configuration."""

    epochs: int = 100
    batch_size: int = 8192
    learning_rate: float = 1e-3
    regularization: float = 1e-5
    embedding_dim: int = 64


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    loss: float
    realized_explicit_proportion: float
    n_updates: int


EpochCallback = Callable[[EpochReport], bool]


def train_bpr(
    model: BPRMatrixFactorization,
    sampler: NegativeSampler,
    config: BPRTrainingConfig,
    *,
    device: str | torch.device = "cpu",
    seed: int,
    on_epoch_end: EpochCallback | None = None,
) -> list[EpochReport]:
    """Train ``model`` and return one report per completed epoch.

    The callback can return ``True`` to stop training early, for example after
    validation performance has not improved for a fixed patience.
    """
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sampler.n_positives == 0:
        raise ValueError("training data contains no positive interactions")

    device_object = torch.device(device)
    model.to(device_object)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(seed)

    user_indices = model.encode_users(sampler.positive_user_ids)
    positive_indices = model.encode_items(sampler.positive_item_ids)
    n_triplets = int(user_indices.size)

    history: list[EpochReport] = []
    for epoch in range(1, config.epochs + 1):
        sampled = sampler.sample()
        negative_indices = model.encode_items(sampled.items)
        order = rng.permutation(n_triplets)

        total_loss = 0.0
        n_updates = 0
        model.train()

        for start in range(0, n_triplets, config.batch_size):
            batch = order[start : start + config.batch_size]
            users = torch.from_numpy(user_indices[batch]).to(device_object)
            positives = torch.from_numpy(positive_indices[batch]).to(device_object)
            negatives = torch.from_numpy(negative_indices[batch]).to(device_object)

            optimizer.zero_grad(set_to_none=True)
            loss = model.bpr_loss(
                users,
                positives,
                negatives,
                regularization=config.regularization,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * int(batch.size)
            n_updates += 1

        report = EpochReport(
            epoch=epoch,
            loss=total_loss / n_triplets,
            realized_explicit_proportion=sampled.realized_explicit_proportion,
            n_updates=n_updates,
        )
        history.append(report)

        if on_epoch_end is not None and on_epoch_end(report):
            break

    return history
