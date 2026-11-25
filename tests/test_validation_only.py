import json
import sys

import pandas as pd
from scripts import train


def test_train_does_not_evaluate_test(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    columns = ["user_id", "movie_id", "rating", "timestamp", "label"]
    pd.DataFrame(
        [(1, 1, 5, 1, "positive"), (2, 2, 5, 1, "positive"), (3, 3, 5, 1, "positive")],
        columns=columns,
    ).to_csv(data / "train.csv", index=False)
    pd.DataFrame([(1, 2, 5, 2, "positive")], columns=columns).to_csv(
        data / "validation.csv", index=False
    )
    pd.DataFrame([(1, 3, 5, 3, "positive")], columns=columns).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"movie_id": [1, 2, 3], "train_positive_count": [1, 1, 1]}).to_csv(
        data / "catalog.csv", index=False
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("fixture: true\n")
    out = tmp_path / "run"
    original = train.evaluate
    original_per_user = train.evaluate_per_user

    def validation_only(*args, **kwargs):
        assert kwargs["period"] == "validation"
        return original(*args, **kwargs)

    def validation_per_user_only(*args, **kwargs):
        assert kwargs["period"] == "validation"
        return original_per_user(*args, **kwargs)

    monkeypatch.setattr(train, "evaluate", validation_only)
    monkeypatch.setattr(train, "evaluate_per_user", validation_per_user_only)
    monkeypatch.setattr(train, "git_commit", lambda root: "fixture")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--model",
            "mf_bpr",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--data-dir",
            str(data),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(out),
        ],
    )
    train.main()
    result = json.loads((out / "metrics.json").read_text())
    assert "test" not in result and "test_per_user" not in result
    assert result["validation_history"] and (out / "best.pt").is_file()
