"""LightGCN++ with flexible degree normalization and adaptive layer pooling.

LightGCN++ (Lee et al., RecSys 2024) keeps LightGCN's linear propagation and
BPR objective but replaces the fixed symmetric normalization with
``D^-alpha A D^-beta``, unit-normalizes embeddings before each propagation step,
and pools the input and propagated layers with a single ``gamma`` weight.
"""

from collections.abc import Sequence

import torch

from src.models.lightgcn import IntArray, LightGCN


class LightGCNPlusPlus(LightGCN):
    """LightGCN variant with tunable degree exponents and layer pooling."""

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
        alpha: float = 0.6,
        beta: float = -0.1,
        gamma: float = 0.2,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        super().__init__(
            user_ids,
            item_ids,
            user_edges,
            item_edges,
            n_layers=n_layers,
            embedding_dim=embedding_dim,
            init_std=init_std,
            alpha=alpha,
            beta=beta,
        )

    def _propagate(self) -> torch.Tensor:
        layers = [self.embedding.weight]
        embeddings = self.embedding.weight

        for _ in range(self.n_layers):
            norm = embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
            embeddings = torch.sparse.mm(self._adjacency, embeddings / norm)
            layers.append(embeddings)

        propagated = torch.stack(layers[1:], dim=0).mean(dim=0)
        return self.gamma * layers[0] + (1.0 - self.gamma) * propagated
