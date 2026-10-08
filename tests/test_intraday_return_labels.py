import argparse

import numpy as np
import pandas as pd
import pytest

from trading_system.experiments.config import ExperimentConfig
from trading_system.labels.config import LabelConfig
from trading_system.labels.intraday_return import build_intraday_return_labels
from trading_system.labels.registry import LabelContext, create_default_label_registry
from trading_system.pipelines.label_arguments import (
    add_label_arguments,
    apply_label_arguments,
    label_config_from_args,
)


def _frame(opens, closes):
    return pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=len(opens)),
         "open": opens, "close": closes}
    )


def test_intraday_labels_use_raw_open_close_and_equality_is_flat():
    frame = _frame([100.0] * 3, [102.0, 99.0, 100.0])
    frame["adj_close"] = [1.0, 1_000.0, 9.0]
    original = frame.copy(deep=True)
    labeled, report = build_intraday_return_labels(frame)
    assert labeled["Label"].tolist() == ["Long", "Short", "Flat"]
    assert labeled["Label_id"].tolist() == [2, 0, 1]
    np.testing.assert_allclose(labeled["intraday_ret"], [0.02, -0.01, 0.0])
    assert labeled["_label_known"].all()
    assert report == {"n_rows": 3, "n_known": 3, "n_unknown": 0,
                      "n_long": 1, "n_flat": 1, "n_short": 1}
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf, -np.inf, "bad"])
@pytest.mark.parametrize("column", ["open", "close"])
def test_invalid_price_is_unknown_not_genuine_flat(invalid, column):
    frame = _frame([100.0, 100.0], [100.0, 101.0])
    frame[column] = frame[column].astype(object)
    frame.loc[1, column] = invalid
    result = create_default_label_registry().generate(frame, LabelConfig.intraday_return())
    assert result.known_mask.tolist() == [True, False]
    assert result.frame.loc[0, "Label"] == "Flat"
    assert np.isnan(result.frame.loc[1, "intraday_ret"])
    assert result.metadata["class_counts"] == {"Short": 0, "Flat": 1, "Long": 0}


def test_intraday_alignment_is_per_ticker_and_last_session_is_known():
    first = _frame([10.0, 20.0, 30.0], [11.0, 19.0, 30.0]).assign(ticker="AAA")
    second = _frame([100.0, 100.0, 100.0], [99.0, 101.0, 102.0]).assign(ticker="BBB")
    frame = pd.concat([first, second], ignore_index=True).sample(frac=1, random_state=3)
    frame["_experiment_split"] = frame["date"].map(
        dict(zip(pd.date_range("2024-01-01", periods=3), ["train", "val", "test"]))
    )
    result = create_default_label_registry().generate(
        frame, LabelConfig.intraday_return(), LabelContext(group_col="ticker")
    )
    assert result.frame["ticker"].tolist() == ["AAA"] * 3 + ["BBB"] * 3
    assert result.frame["Label_id"].tolist() == [2, 0, 1, 0, 2, 2]
    assert result.known_mask.all()
    assert result.class_names == ("Short", "Flat", "Long")
    assert result.semantics == "target_position"
    assert result.metadata["target_available_at"] == "close J"
    assert "execution timing is unchanged" in result.metadata["execution_note"]


def test_intraday_honors_existing_unknown_mask():
    frame = _frame([100.0, 100.0], [101.0, 99.0])
    frame["_label_known"] = [True, False]
    labeled, report = build_intraday_return_labels(frame)
    assert labeled["_label_known"].tolist() == [True, False]
    assert labeled["Label_id"].tolist() == [2, 1]
    assert np.isnan(labeled.loc[1, "label_score"])
    assert report["n_unknown"] == 1


@pytest.mark.parametrize("failure", ["missing_open", "missing_close", "date", "duplicate"])
def test_intraday_rejects_invalid_schema_and_duplicate_dates(failure):
    frame = _frame([100.0, 100.0], [101.0, 99.0])
    if failure.startswith("missing_"):
        frame = frame.drop(columns=failure.removeprefix("missing_"))
    elif failure == "date":
        frame.loc[0, "date"] = pd.NaT
    else:
        frame.loc[1, "date"] = frame.loc[0, "date"]
    with pytest.raises(ValueError):
        build_intraday_return_labels(frame)


@pytest.mark.parametrize("method", ["intraday-return", "intraday_return"])
def test_shared_cli_resolves_intraday_target(method):
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    args = parser.parse_args(["--label-method", method])
    config = apply_label_arguments(ExperimentConfig(), args)
    assert config.label_mode == "intraday_return"
    assert config.resolved_label_config() == LabelConfig.intraday_return()
    assert label_config_from_args(args) == LabelConfig.intraday_return()
    assert config.resolved_class_names() == ("Short", "Flat", "Long")
    assert config.resolved_label_semantics() == "target_position"


@pytest.mark.parametrize("option,value", [
    ("--label-horizon", "10"), ("--label-window", "20"),
    ("--label-buy-threshold", "0.1"), ("--label-cost-bps", "5"),
])
def test_intraday_cli_rejects_parameters_of_other_methods(option, value):
    parser = argparse.ArgumentParser()
    add_label_arguments(parser)
    args = parser.parse_args(["--label-method", "intraday-return", option, value])
    with pytest.raises(ValueError):
        apply_label_arguments(ExperimentConfig(), args)


def test_intraday_registry_rejects_wrong_semantics_and_unknown_parameters():
    registry = create_default_label_registry()
    frame = _frame([100.0], [101.0])
    with pytest.raises(ValueError, match="semantics"):
        registry.generate(frame, LabelConfig(method="intraday_return", semantics="action"))
    with pytest.raises(ValueError, match="parameters"):
        registry.generate(frame, LabelConfig(
            method="intraday_return", semantics="target_position", parameters={"horizon": 5}
        ))
