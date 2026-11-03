# recommender-from-scratch

From-scratch comparison of BPR matrix factorization, LightGCN and a LightGCN++
extension on MovieLens-1M, with a most-popular baseline and three target
explicit-negative proportions.

## Setup

```bash
uv sync
uv run pre-commit install
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
experiment grid. Rerun both scripts after changing the raw data or the protocol
constants; do not edit generated files by hand.

## Training

```bash
uv run python -m scripts.run_popularity
for model in mf_bpr lightgcn_bpr lightgcnpp_bpr; do
  for proportion in 0.0 0.25 0.5; do
    for seed in 42 43 44; do
      uv run python -m scripts.train --model "$model" \
        --explicit-negative-proportion "$proportion" --seed "$seed"
    done
  done
done
```

LightGCN++ uses the paper's recommended defaults `--alpha 0.6 --beta -0.1
--gamma 0.2`; override them for validation-based tuning. Runs write checkpoints,
per-epoch metrics and per-user validation/test metrics under `runs/`, which is
ignored by git. Checkpoint selection uses validation NDCG@20 only. The realized
explicit-negative proportion is recorded in each run's `metrics.json` alongside
the target.

## Analysis and reports

```bash
uv run python -m scripts.analyze_results
```

This writes `docs/analysis.json` with seed aggregates, paired bootstrap
intervals and Holm-corrected Wilcoxon tests for the prespecified comparisons,
plus exploratory LightGCN++ comparisons. See `docs/lightgcn.md` for the graph
model and `docs/results.md` for the stage-2 comparison.
