# Corrected experiment workflow

## Validation-only runs

Install the locked environment, obtain the fixed MovieLens-1M release and prepare
the frozen split as described in the README. Commit the reviewed code before
training so the saved Git revision identifies the experiment implementation.

```bash
uv sync --frozen
uv run python -m scripts.prepare_data
uv run python -m scripts.train --model lightgcn_bpr --seed 42 \
  --explicit-negative-proportion 0.0 --regularization 1e-5
```

Training now evaluates validation only. Each default directory includes a SHA-256
of the CLI, resolved device, data fingerprints, manifest and Git revision. Explicit
`--output-dir` is supported for selected configurations; it must not exist. Old
checkpoints and metrics are never moved or overwritten. A failed run should be
inspected and a new directory chosen for the retry.

MF, LightGCN and LightGCN++ use initial-embedding L2 by default. Use
`--regularization-on propagated` only for a labelled legacy graph ablation.
Selection remains validation NDCG@20. The 25% condition is sensitivity; LightGCN++
is exploratory. Do not select models or hyperparameters using test results.

## Final controlled comparison

Use a separate fresh directory for selected configurations. The following example
uses the existing controlled settings (100 epochs, 64 dimensions, learning rate
0.001, L2 0.00001, batch size 8192, two graph layers, patience 10). All models use
the same training settings; this is not a fair-tuning search. Run 0% and 50% first;
include 25% only as the prespecified sensitivity check. Save validation results and
record selection decisions before running the final evaluation loop.

```bash
for proportion in 0.0 0.5 0.25; do
  case "$proportion" in 0.0) percent=0;; 0.5) percent=50;; 0.25) percent=25;; esac
  for model in mf_bpr lightgcn_bpr; do
    for seed in 42 43 44; do
      uv run python -m scripts.train --model "$model" \
        --explicit-negative-proportion "$proportion" --seed "$seed" \
        --output-dir "runs/corrected/${model}_p${percent}/seed_${seed}"
    done
  done
done
```

Once the configurations are locked, explicitly evaluate each selected checkpoint.
This refuses repeat evaluation and checks the checkpoint and data fingerprints.

```bash
for run in runs/corrected/*/seed_*; do
  uv run python -m scripts.evaluate_checkpoint --run-dir "$run"
done
uv run python -m scripts.run_popularity --output runs/corrected/popularity/metrics.json
uv run python -m scripts.analyze_results --runs-dir runs/corrected \
  --output docs/final-analysis.json --table-output docs/final-tables.md
```

For tuned runs, use distinct directories during tuning and copy only the locked,
selected seed directories into a new analysis directory using the same naming
scheme. Do not combine validation and final data or retrain on validation.
The analyzer rejects missing seed metadata, legacy L2, missing test results,
nonmatching user sets and non-773-user cohorts. Add `--include-lightgcnpp` only
when all exploratory configurations are explicitly selected and evaluated.
The existing `docs/analysis.json` and `docs/results.md` remain historical legacy
artifacts. No updated quantitative claim is valid until corrected runs finish.

The paired intervals bootstrap the 773 users after averaging each user's metric
across fixed seeds (10,000 replicates); they describe user-level uncertainty,
not uncertainty over new seeds or other datasets. Across-seed SD uses `ddof=1`.
The Wilcoxon test uses the existing two-sided normal approximation with tied-rank
variance correction and zero differences removed; Holm adjusts the four
prespecified primary comparisons. Gains over Most Popular are absolute differences.

## Ranking export and popularity

```bash
mkdir -p runs/corrected/exports
uv run python -m scripts.export_rankings --popularity \
  --output runs/corrected/exports/popularity.jsonl
for run in runs/corrected/*/seed_*; do
  name="$(basename "$(dirname "$run")")_$(basename "$run")"
  uv run python -m scripts.export_rankings --run-dir "$run" \
    --output "runs/corrected/exports/${name}.jsonl"
done
uv run python -m scripts.analyze_popularity runs/corrected/exports/*.jsonl \
  --output runs/corrected/popularity-analysis.json
```

Each JSONL record contains one eligible user's top 20 plus full ranks of their
eligible test likes and applicable test dislikes. Candidate IDs, scores and
masking come from the same iterator used by `evaluate_per_user`; equal scores
are ordered by ascending movie ID. Titles are read from the original `movies.dat`.
Legacy checkpoints may be exported with `--allow-legacy` but are explicitly
labelled propagated-L2 and cannot be assumed to have verified data provenance.

Popularity is the count of positive training ratings per catalog item. Freeze
four count bins before new test runs: 1–9, 10–99, 100–999, 1000+. The log-like
thresholds cover the manifest's 1–2749 range without using held-out interactions.
For each bin, exposure is the macro-average fraction of each user's returned
top-20 list in that bin. Recall is the macro-average fraction of a user's eligible
positive targets in that bin retrieved at rank 20 or earlier, over users having
at least one such target. An empty target bin has undefined (null) recall.
Report catalog item counts, eligible user counts and target counts alongside both
metrics. Compare only exports with identical user, target and candidate signatures.

## Verification

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src scripts
uv run pytest -q
```

Run the full experiments separately on suitable hardware. Neither training data
nor checkpoints belong in Git. The old report must not attribute a causal graph
effect to the legacy propagated-L2 comparison.
