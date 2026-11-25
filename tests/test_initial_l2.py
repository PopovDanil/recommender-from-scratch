import numpy as np
import pytest
import torch
from src.models.bpr_mf import BPRMatrixFactorization
from src.models.lightgcn import LightGCN
from src.models.lightgcnpp import LightGCNPlusPlus


@pytest.mark.parametrize("model_type", [LightGCN, LightGCNPlusPlus])
def test_initial_l2_matches_mf_and_propagated_scores(model_type):
    users = [1, 2]
    items = [10, 20]
    edges_u = np.array([1, 1, 2], dtype=np.int64)
    edges_i = np.array([10, 20, 20], dtype=np.int64)
    model = model_type(users, items, edges_u, edges_i, embedding_dim=2)
    mf = BPRMatrixFactorization(users, items, embedding_dim=2)
    legacy = model_type(
        users, items, edges_u, edges_i, embedding_dim=2, regularization_on="propagated"
    )
    weights = torch.tensor([[1.0, 2.0], [2.0, 1.0], [0.0, 2.0], [3.0, 1.0]])
    with torch.no_grad():
        model.embedding.weight.copy_(weights)
        legacy.embedding.weight.copy_(weights)
        mf.user_embedding.weight.copy_(weights[:2])
        mf.item_embedding.weight.copy_(weights[2:])
    u, p, n = torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([3, 2])
    coefficient = 0.01
    manual = (
        coefficient
        * (
            weights[u].square().sum(1) + weights[p].square().sum(1) + weights[n].square().sum(1)
        ).mean()
    )
    base = model.bpr_loss(u, p, n)
    assert torch.allclose(model.bpr_loss(u, p, n, regularization=coefficient) - base, manual)
    mf_base = mf.bpr_loss(u, p - 2, n - 2)
    assert torch.allclose(
        mf.bpr_loss(u, p - 2, n - 2, regularization=coefficient) - mf_base, manual
    )
    propagated = model._propagate()
    delta = (propagated[u] * (propagated[p] - propagated[n])).sum(1)
    assert torch.allclose(base, -torch.nn.functional.logsigmoid(delta).mean())
    assert not torch.allclose(propagated, weights)
    assert torch.allclose(legacy.bpr_loss(u, p, n), base)
    legacy_penalty = (
        coefficient
        * (
            propagated[u].square().sum(1)
            + propagated[p].square().sum(1)
            + propagated[n].square().sum(1)
        ).mean()
    )
    assert torch.allclose(
        legacy.bpr_loss(u, p, n, regularization=coefficient) - base, legacy_penalty
    )
    model.eval()
    assert np.allclose(model.score(1, items), (propagated[2:] @ propagated[0]).detach().numpy())
