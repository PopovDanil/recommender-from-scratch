# recommender-from-scratch

From-scratch comparison of BPR matrix factorization and LightGCN on MovieLens-1M,
with a most-popular baseline and three target explicit-negative proportions.

## Setup

```bash
uv sync
```

## Data preparation

Download the fixed MovieLens-1M release from GroupLens and extract it to
`data/ml-1m/`. Raw data, generated splits and run artifacts are not committed.

```bash
uv run python -m scripts.eda_movielens   # writes docs/data-audit.md
uv run python -m scripts.prepare_data    # writes data/processed/ml-1m and data/manifest.yaml
```

The temporal boundaries are fixed in `src/data/splitting.py`. The manifest records
the exact boundaries, period counts, catalog checksum, negative-pool audit and the
six-configuration experiment grid. Rerun both scripts after changing the raw data
or the protocol constants; do not edit generated files by hand.

## Training

```bash
uv run python -m scripts.run_popularity
for proportion in 0.0 0.25 0.5; do
  for seed in 42 43 44; do
    uv run python -m scripts.train_bpr \
      --explicit-negative-proportion "$proportion" --seed "$seed"
  done
done
```

Runs write checkpoints and per-epoch metrics under `runs/`, which is ignored by git.
Checkpoint selection uses validation NDCG@20 only. The realized explicit-negative
proportion is recorded in each run's `metrics.json` alongside the target.
