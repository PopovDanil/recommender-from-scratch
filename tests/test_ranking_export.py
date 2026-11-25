import pandas as pd
import pytest
from scripts.analyze_popularity import summarize
from scripts.export_rankings import export_rows
from src.data.processed import ProcessedSplits
from src.evaluation.metrics import DEFAULT_METRICS, EvaluationSplit, evaluate_per_user
from src.models.popularity import Popularity


def test_export_reproduces_canonical_metrics():
    train = pd.DataFrame(
        [
            (1, 1, 5, "positive"),
            (2, 2, 5, "positive"),
            (2, 3, 5, "positive"),
            (2, 4, 5, "positive"),
        ],
        columns=["user_id", "movie_id", "rating", "label"],
    )
    validation = pd.DataFrame([(1, 2, 3, "neutral")], columns=train.columns)
    test = pd.DataFrame([(1, 3, 5, "positive"), (1, 4, 1, "negative")], columns=train.columns)
    catalog = pd.DataFrame({"movie_id": [1, 2, 3, 4], "train_positive_count": [1, 1, 1, 1]})
    splits = ProcessedSplits(train, validation, test, catalog)
    model = Popularity.from_train(train, splits.catalog_ids)
    titles = {i: str(i) for i in range(1, 5)}
    rows = export_rows(splits, model, titles, {"experiment": "popularity"})
    (row,) = rows
    assert row["candidate_set_size"] == 2
    assert [entry["movie_id"] for entry in row["top20"]] == [3, 4]
    assert [entry["test_label"] for entry in row["top20"]] == ["positive", "negative"]
    ranked = [entry["movie_id"] for entry in row["top20"]]
    relevant = {
        entry["movie_id"] for entry in row["test_items"] if entry["test_label"] == "positive"
    }
    expected = evaluate_per_user(
        EvaluationSplit(train, validation, test, catalog), model, period="test"
    )
    for metric in DEFAULT_METRICS:
        assert metric.function(ranked, relevant, metric.k) == pytest.approx(
            expected[1][metric.name]
        )
    stats = summarize(rows, {1: 1, 2: 1, 3: 1, 4: 1})
    assert stats["1-9"]["exposure@20"] == 1
    assert stats["1-9"]["recall@20"] == 1
    assert stats["10-99"]["recall@20"] is None
