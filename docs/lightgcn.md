# LightGCN + BPR

Implementation notes for the graph recommender in `src/models/lightgcn.py` and its
training entry point in `scripts/train.py`.

## Graph construction

The user-item bipartite graph contains only positive training interactions on the
shared catalog (`positive_interaction_edges`). Neutral ratings, explicit negatives and
all validation/test interactions are excluded. Each catalog-positive edge contributes
both directions to the node matrix.

Nodes are indexed as users `0..n_users-1` followed by items `n_users..n_nodes-1`.
The adjacency is symmetrically normalized as

```text
A_hat[u, i] = A[u, i] / sqrt(deg(u) * deg(i))
```

and stored as a coalesced sparse tensor. The sparse tensor is a non-persistent buffer,
so checkpoints contain only the embedding table and can be rebuilt from the frozen
split with the current catalog.

## Propagation and scoring

With initial node embeddings `E^(0)`, LightGCN uses the linear propagation rule

```text
E^(k+1) = A_hat E^(k)
E_final = mean(E^(0), E^(1), ..., E^(K))
```

There are no feature-transformation or nonlinearity matrices. The default is `K = 2`
layers; `--n-layers` changes it. A user and an item are scored by the dot product of
their final representations. The evaluator sees the same `Recommender.score` contract
as for MF, so masking and full-catalog ranking are identical.

## Training objective

LightGCN is trained with the same BPR pairwise loss and negative sampler as MF + BPR:

```text
L = -mean(log sigmoid(x_ui - x_uj)) + lambda * mean(||e_u||^2 + ||e_i||^2 + ||e_j||^2)
```

The regularization term uses the propagated triplets, matching the MF-BPR form. The
negative sampler, number of triplets, batch size, optimizer, learning rate and early
stopping rule are shared with the MF runs, keeping the training budget comparable.
The training loop is model-agnostic in `src/training/bpr.py`.

## Run protocol

```bash
uv run python -m scripts.train --model lightgcn_bpr \
  --explicit-negative-proportion 0.0 --seed 42
```

Each run writes `best.pt` and `metrics.json` under
`runs/lightgcn_bpr_p<percent>/seed_<seed>/`. Checkpoint selection uses validation
NDCG@20 only. The script also stores per-user validation and test metrics for the
paired statistical analysis in `scripts/analyze_results.py`.

## Avoided information leaks

- The graph is rebuilt from training positives only and is never updated with
  validation or test edges.
- Ranking masks follow the shared evaluator: train-rated items at validation, and
  train- plus validation-rated items at test.
- No retraining on train + validation is performed before test evaluation.
