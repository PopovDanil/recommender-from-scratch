"""BPR matrix factorization over user and item embeddings."""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from src.models.base import Recommender

IntArray = NDArray[np.int64]


class BPRMatrixFactorization(nn.Module, Recommender):
    """Dot-product recommender trained with the pairwise BPR objective."""

    def __init__(
        self,
        user_ids: Sequence[int],
        item_ids: Sequence[int],
        *,
        embedding_dim: int = 64,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self._user_index = {int(user): index for index, user in enumerate(user_ids)}
        self._item_index = {int(item): index for index, item in enumerate(item_ids)}
        self.user_embedding = nn.Embedding(len(self._user_index), embedding_dim)
        self.item_embedding = nn.Embedding(len(self._item_index), embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=init_std)
        nn.init.normal_(self.item_embedding.weight, std=init_std)

    def bpr_loss(
        self,
        user_indices: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor,
        *,
        regularization: float = 0.0,
    ) -> torch.Tensor:
        users = self.user_embedding(user_indices)
        positives = self.item_embedding(positive_indices)
        negatives = self.item_embedding(negative_indices)

        positive_scores = (users * positives).sum(dim=1)
        negative_scores = (users * negatives).sum(dim=1)
        loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()

        if regularization > 0.0:
            norms = (
                users.pow(2).sum(dim=1) + positives.pow(2).sum(dim=1) + negatives.pow(2).sum(dim=1)
            )
            loss = loss + regularization * norms.mean()

        return loss

    def encode_users(self, user_ids: IntArray) -> IntArray:
        return np.array(
            [self._user_index[int(user)] for user in user_ids],
            dtype=np.int64,
        )

    def encode_items(self, item_ids: IntArray) -> IntArray:
        return np.array(
            [self._item_index[int(item)] for item in item_ids],
            dtype=np.int64,
        )

    def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        user_position = self._user_index.get(int(user_id))
        if user_position is None:
            return [0.0 for _ in item_ids]

        positions = np.fromiter(
            (self._item_index.get(int(item), -1) for item in item_ids),
            dtype=np.int64,
            count=len(item_ids),
        )
        scores = np.zeros(len(item_ids), dtype=np.float64)
        known = positions >= 0
        if known.any():
            with torch.no_grad():
                user = self.user_embedding.weight[user_position]
                items = self.item_embedding.weight[torch.from_numpy(positions[known])]
                scores[known] = (items @ user).cpu().numpy()
        return scores.tolist()
