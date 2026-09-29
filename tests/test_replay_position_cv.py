"""Frozen replay must select only the nine Sharpe N0 checkpoints."""

import json

import pandas as pd
import pytest

from trading_system.experiments import replay_position_cv as replay


def test_replay_reference_filters_candidate_and_refuses_overwrite(monkeypatch, tmp_path):
    cv = tmp_path / "cv"
    cv.mkdir()
    rows = [
        {"objective": "sharpe", "status": "ok", "parameters": {"temporal_pooling": "attention"},
         "seed": seed, "fold": fold, "candidate": "n0"}
        for seed in (1, 7, 19) for fold in (0, 1, 2)
    ]
    rows.append({"objective": "sharpe", "status": "ok", "parameters": {
        "temporal_pooling": "attention", "input_normalization": "revin"},
        "seed": 1, "fold": 0, "candidate": "revin"})
    (cv / "folds.json").write_text(json.dumps(rows))
    monkeypatch.setattr(replay, "read_parquet_dataset", lambda _: pd.DataFrame({"ticker": ["A"]}))
    monkeypatch.setattr(replay, "prepare_feature_sources", lambda frame, **kwargs: (frame, {}))
    seen = []

    def fake_fold(frame, row):
        seen.append((row["seed"], row["fold"]))
        return pd.DataFrame({"position": [0.1]}), {"regularized_sharpe": 0.5}

    monkeypatch.setattr(replay, "replay_fold", fake_fold)
    output = tmp_path / "replay.parquet"
    report = replay.replay_reference(cv, tmp_path / "source.parquet", ["A"], output)
    assert report["runs"] == 9
    assert len(seen) == 9
    assert len(pd.read_parquet(output)) == 9
    with pytest.raises(FileExistsError):
        replay.replay_reference(cv, tmp_path / "source.parquet", ["A"], output)
