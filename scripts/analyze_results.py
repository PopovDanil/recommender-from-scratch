"""Aggregate stage-2 runs and run the prespecified paired comparisons.

The script reads per-seed ``metrics.json`` files under ``runs/`` and writes
``results/stage2/analysis.json``. Primary inference uses test NDCG@20, with
seed-averaged user metrics, 10,000 paired bootstrap resamples and a two-sided
Wilcoxon signed-rank test followed by Holm correction for the four confirmatory
comparisons. The 25% conditions are reported as sensitivity analyses.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from src.experiments import Experiment, experiment_grid

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20")
PRIMARY_METRIC = "ndcg@20"
PERIODS = ("validation", "test")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20241102


@dataclass(frozen=True)
class Comparison:
    name: str
    left: str
    right: str


PRIMARY_COMPARISONS = (
    Comparison("lightgcn_vs_mf_p0", "lightgcn_bpr_p0", "mf_bpr_p0"),
    Comparison("lightgcn_vs_mf_p50", "lightgcn_bpr_p50", "mf_bpr_p50"),
    Comparison("mf_p50_vs_p0", "mf_bpr_p50", "mf_bpr_p0"),
    Comparison("lightgcn_p50_vs_p0", "lightgcn_bpr_p50", "lightgcn_bpr_p0"),
)


def _run_path(runs_dir: Path, experiment: str, seed: int) -> Path:
    return runs_dir / experiment / f"seed_{seed}" / "metrics.json"


def _load_run(runs_dir: Path, experiment: str, seed: int) -> dict[str, Any]:
    path = _run_path(runs_dir, experiment, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing run metrics: {path}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _aggregate(runs_dir: Path, experiment: Experiment) -> dict[str, Any]:
    runs = [_load_run(runs_dir, experiment.name, seed) for seed in experiment.seeds]
    summary: dict[str, Any] = {
        "experiment": experiment.name,
        "model": experiment.model,
        "target_explicit_negative_proportion": experiment.explicit_negative_proportion,
        "seeds": list(experiment.seeds),
        "realized_explicit_negative_proportion": {
            "mean": float(np.mean([run["realized_explicit_negative_proportion"] for run in runs])),
            "std": float(
                np.std([run["realized_explicit_negative_proportion"] for run in runs], ddof=1)
            ),
        },
        "best_epoch": {
            "mean": float(np.mean([run["best_epoch"] for run in runs])),
            "values": [int(run["best_epoch"]) for run in runs],
        },
    }

    for period in PERIODS:
        summary[period] = {
            metric: {
                "mean": float(np.mean([run[period][metric] for run in runs])),
                "std": float(np.std([run[period][metric] for run in runs], ddof=1)),
            }
            for metric in METRICS
        }

    return summary


def _seed_averaged_users(
    runs_dir: Path,
    experiment: Experiment,
    *,
    period: str,
    metric: str,
) -> dict[int, float]:
    samples: dict[int, list[float]] = {}
    for seed in experiment.seeds:
        run = _load_run(runs_dir, experiment.name, seed)
        per_user = run[f"{period}_per_user"]
        for user_id, values in per_user.items():
            samples.setdefault(int(user_id), []).append(float(values[metric]))

    missing = [user for user, values in samples.items() if len(values) != len(experiment.seeds)]
    if missing:
        raise ValueError(f"{experiment.name} has users missing seed metrics: {missing[:5]}")
    return {user: float(np.mean(values)) for user, values in samples.items()}


def _paired_bootstrap(
    left: dict[int, float],
    right: dict[int, float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    users = sorted(set(left) & set(right))
    if not users:
        raise ValueError("paired comparison has no common users")

    differences = np.array([left[user] - right[user] for user in users], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_users = differences.size
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.integers(0, n_users, size=n_users)
        bootstrap_means[index] = float(differences[sample].mean())

    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "n_users": float(n_users),
        "mean_difference": float(differences.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def _wilcoxon_signed_rank(left: dict[int, float], right: dict[int, float]) -> dict[str, float]:
    users = sorted(set(left) & set(right))
    differences = np.array([left[user] - right[user] for user in users], dtype=np.float64)
    nonzero = differences[differences != 0.0]
    n = nonzero.size
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_nonzero": 0.0}

    ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
    positive_rank = float(ranks[nonzero > 0].sum())

    mean = n * (n + 1) / 4.0
    tie_counts = pd.Series(np.abs(nonzero)).value_counts()
    tie_correction = float(sum(count**3 - count for count in tie_counts if count > 1))
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0
    if variance <= 0.0:
        return {"statistic": positive_rank, "p_value": 1.0, "n_nonzero": float(n)}

    z_score = (positive_rank - mean) / math.sqrt(variance)
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return {
        "statistic": positive_rank,
        "p_value": min(1.0, p_value),
        "n_nonzero": float(n),
    }


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[name] = running
    return adjusted


def _user_activity_groups(
    train: pd.DataFrame,
    users: dict[int, float],
) -> dict[str, dict[str, float]]:
    positives = train[train["label"] == "positive"]
    counts = positives.groupby("user_id").size()
    user_counts = pd.Series({user: int(counts.get(user, 0)) for user in users})
    terciles = np.quantile(user_counts.to_numpy(), [1 / 3, 2 / 3])
    groups = {
        "low": user_counts <= terciles[0],
        "medium": (user_counts > terciles[0]) & (user_counts <= terciles[1]),
        "high": user_counts > terciles[1],
    }
    return {
        name: {
            "n_users": float(mask.sum()),
            "ndcg@20": float(np.mean([users[user] for user in mask[mask].index])),
        }
        for name, mask in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "ml-1m",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "stage2" / "analysis.json",
    )
    args = parser.parse_args()

    experiments = experiment_grid()
    aggregates = {
        experiment.name: _aggregate(args.runs_dir, experiment) for experiment in experiments
    }

    per_user = {
        experiment.name: _seed_averaged_users(
            args.runs_dir,
            experiment,
            period="test",
            metric=PRIMARY_METRIC,
        )
        for experiment in experiments
    }

    comparisons: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for comparison in PRIMARY_COMPARISONS:
        left = per_user[comparison.left]
        right = per_user[comparison.right]
        bootstrap = _paired_bootstrap(left, right)
        wilcoxon = _wilcoxon_signed_rank(left, right)
        comparisons[comparison.name] = {
            "left": comparison.left,
            "right": comparison.right,
            "metric": PRIMARY_METRIC,
            "bootstrap": bootstrap,
            "wilcoxon": wilcoxon,
        }
        raw_p_values[comparison.name] = wilcoxon["p_value"]

    adjusted = _holm(raw_p_values)
    for name, adjusted_p in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = adjusted_p

    popularity_path = args.runs_dir / "popularity" / "metrics.json"
    popularity = (
        json.loads(popularity_path.read_text(encoding="utf-8"))
        if popularity_path.is_file()
        else None
    )

    train = pd.read_csv(args.data_dir / "train.csv")
    activity = {
        experiment.name: _user_activity_groups(train, per_user[experiment.name])
        for experiment in experiments
    }

    analysis = {
        "primary_metric": PRIMARY_METRIC,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "aggregates": aggregates,
        "comparisons": comparisons,
        "popularity": popularity,
        "test_user_activity_terciles": activity,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print(f"wrote {args.output}")
    for name, result in comparisons.items():
        bootstrap = result["bootstrap"]
        print(
            f"{name}: diff={bootstrap['mean_difference']:+.4f} "
            f"CI=[{bootstrap['ci95_lower']:+.4f}, {bootstrap['ci95_upper']:+.4f}] "
            f"wilcoxon_p={result['wilcoxon']['p_value']:.4f} "
            f"holm_p={result['holm_adjusted_p']:.4f}"
        )
    for name, summary in aggregates.items():
        test = summary["test"][PRIMARY_METRIC]
        print(f"{name}: {PRIMARY_METRIC}={test['mean']:.4f} +/- {test['std']:.4f}")


if __name__ == "__main__":
    main()
