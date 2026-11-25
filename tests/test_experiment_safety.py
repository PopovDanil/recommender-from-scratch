import json

import pytest
from scripts.analyze_results import _load_run, _paired_bootstrap, _wilcoxon_signed_rank
from src.evaluation.paired import exact_users
from src.training.artifacts import reserve_run, run_id


def test_run_identity_and_collision(tmp_path):
    original = {"regularization": 1e-5, "layers": 2, "alpha": 0.6, "beta": -0.1, "gamma": 0.2}
    for key in original:
        changed = dict(original)
        changed[key] += 1
        assert run_id(changed) != run_id(original)
    assert run_id(dict(reversed(list(original.items())))) == run_id(original)
    path = tmp_path / "run"
    reserve_run(path)
    (path / "best.pt").write_text("original")
    with pytest.raises(FileExistsError):
        reserve_run(path)
    assert (path / "best.pt").read_text() == "original"


def test_exact_users():
    users = set(range(773))
    assert exact_users(users, users) == sorted(users)
    with pytest.raises(ValueError, match=r"missing IDs=\[42\]"):
        exact_users(users, users - {42})
    with pytest.raises(ValueError, match=r"extra IDs=\[900\]"):
        exact_users(users, users | {900})
    for method in (_paired_bootstrap, _wilcoxon_signed_rank):
        with pytest.raises(ValueError, match="missing IDs"):
            method({1: 0.2, 2: 0.3}, {1: 0.2})


def test_seed_mismatch(tmp_path):
    path = tmp_path / "mf_bpr_p0" / "seed_42"
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({"seed": 43, "experiment": "mf_bpr_p0"}))
    with pytest.raises(ValueError, match="mismatched seed"):
        _load_run(tmp_path, "mf_bpr_p0", 42)
