import json

import pytest

from scripts.download_us_benchmark_data import build_parser, load_tickers


def test_tracked_us_downloader_reads_frozen_selection(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"tickers": ["AAPL", "MSFT"]}))
    assert load_tickers(selection) == ("AAPL", "MSFT")
    args = build_parser().parse_args([])
    assert args.start == "2005-01-03"
    assert args.clean_output.name == "mt5_stocks_us_daily_clean.parquet"


def test_tracked_us_downloader_rejects_duplicate_selection(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"tickers": ["AAPL", "AAPL"]}))
    with pytest.raises(ValueError, match="unique"):
        load_tickers(selection)
