"""LightGCN graph recommender trained with the pairwise BPR objective.

The bipartite user-item graph is built from training positives only. Embeddings
propagate over the symmetrically normalized adjacency and the layer outputs,
including the input layer, are averaged to form the final representations.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from src.data.splitting import POSITIVE
from src.models.base import Recommender

IntArray = NDArray[np.int64]


def positive_interaction_edges(
    train_ratings: pd.DataFrame,
    item_ids: Sequence[int],
) -> tuple[IntArray, IntArray]:
    """Return user and item ids for catalog-positive training interactions.

    Only rows labelled positive whose movie belongs to the shared catalog are
    kept, so the graph never sees neutral or explicit-negative ratings.
    """
    catalog = {int(item) for item in item_ids}
    positives = train_ratings[
        (train_ratings["label"] == POSITIVE) & train_ratings["movie_id"].isin(catalog)
    ]
    users = np.asarray(positives["user_id"].to_numpy(), dtype=np.int64)
    items = np.asarray(positives["movie_id"].to_numpy(), dtype=np.int64)
    return users, items


def _normalized_adjacency(
    n_users: int,
    n_items: int,
    user_positions: IntArray,
    item_positions: IntArray,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> torch.Tensor:
    n_nodes = n_users + n_items
    item_nodes = item_positions + n_users
    rows = np.concatenate([user_positions, item_nodes])
    cols = np.concatenate([item_nodes, user_positions])
    degrees = np.bincount(rows, minlength=n_nodes).astype(np.float64)
    values = np.power(degrees[rows], -alpha) * np.power(degrees[cols], -beta)

    indices = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    weights = torch.from_numpy(values.astype(np.float32))
    with torch.sparse.check_sparse_tensor_invariants(enable=False):  # type: ignore[no-untyped-call]
        return torch.sparse_coo_tensor(indices, weights, (n_nodes, n_nodes)).coalesce()


class LightGCN(nn.Module, Recommender):
    """Light graph convolution over a normalized user-item bipartite graph."""

    def __init__(
        self,
        user_ids: Sequence[int],
        item_ids: Sequence[int],
        user_edges: IntArray,
        item_edges: IntArray,
        *,
        n_layers: int = 2,
        embedding_dim: int = 64,
        init_std: float = 0.1,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if user_edges.size != item_edges.size:
            raise ValueError("user and item edge arrays must have the same length")
        if user_edges.size == 0:
            raise ValueError("LightGCN requires at least one training positive edge")

        self._user_index = {int(user): index for index, user in enumerate(user_ids)}
        self._item_index = {int(item): index for index, item in enumerate(item_ids)}
        self.n_layers = n_layers

        user_positions = np.array(
            [self._user_index[int(user)] for user in user_edges],
            dtype=np.int64,
        )
        item_positions = np.array(
            [self._item_index[int(item)] for item in item_edges],
            dtype=np.int64,
        )

        self.embedding = nn.Embedding(len(self._user_index) + len(self._item_index), embedding_dim)
        nn.init.normal_(self.embedding.weight, std=init_std)

        adjacency = _normalized_adjacency(
            len(self._user_index),
            len(self._item_index),
            user_positions,
            item_positions,
            alpha=alpha,
            beta=beta,
        )
        self.register_buffer("_adjacency", adjacency, persistent=False)
        self._combined_cache: torch.Tensor | None = None
        self.register_load_state_dict_post_hook(LightGCN._clear_cache)  # type: ignore[no-untyped-call]

    @staticmethod
    def _clear_cache(module: nn.Module, _incompatible_keys: object) -> None:
        if isinstance(module, LightGCN):
            module._combined_cache = None

    def train(self, mode: bool = True) -> "LightGCN":
        self._combined_cache = None
        super().train(mode)
        return self

    def _propagate(self) -> torch.Tensor:
        layers = [self.embedding.weight]
        embeddings = self.embedding.weight
        for _ in range(self.n_layers):
            embeddings = torch.sparse.mm(self._adjacency, embeddings)
            layers.append(embeddings)
        return torch.stack(layers, dim=0).mean(dim=0)

    def _combined_embeddings(self) -> torch.Tensor:
        if self._combined_cache is None:
            with torch.no_grad():
                self._combined_cache = self._propagate()
        return self._combined_cache

    def bpr_loss(
        self,
        user_indices: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor,
        *,
        regularization: float = 0.0,
    ) -> torch.Tensor:
        embeddings = self._propagate()
        users = embeddings[user_indices]
        positives = embeddings[positive_indices]
        negatives = embeddings[negative_indices]

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
        offset = len(self._user_index)
        return np.array(
            [offset + self._item_index[int(item)] for item in item_ids],
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
            embeddings = self._combined_embeddings()
            offset = len(self._user_index)
            items = embeddings[torch.from_numpy(positions[known] + offset)]
            user = embeddings[user_position]
            scores[known] = (items @ user).cpu().numpy()
        return scores.tolist()
