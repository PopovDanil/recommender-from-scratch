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

## Training and analysis

See [the experiment workflow](docs/experiment-workflow.md) for collision-safe
validation runs, explicit final evaluation, ranking export and popularity analysis.
The old numbers in `docs/results.md` and `docs/analysis.json` describe legacy runs
and must not be interpreted as corrected initial-L2 LightGCN results.
