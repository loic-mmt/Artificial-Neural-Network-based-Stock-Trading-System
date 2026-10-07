"""Small offline CSV fixtures exercise FNSPID provenance and strict parsing."""

import csv
import json

import pandas as pd
import pytest

from trading_system.data.fnspid_import import import_fnspid
from trading_system.data.news_collection import _output_lock


HEADER = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Lsa_summary"]


def _csv(tmp_path, name, rows, header=HEADER):
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _info(tmp_path, revision="a" * 40):
    path = tmp_path / "dataset-info.json"
    path.write_text(json.dumps({"id": "Zdong104/FNSPID", "sha": revision}), encoding="utf-8")
    return path


def _import(tmp_path, paths, **kwargs):
    arguments = {"tickers": ["AAPL", "JPM"], "start": "2020-01-01", "end": "2020-04-01",
                 "dataset_info": kwargs.get("dataset_info") or _info(tmp_path), "chunksize": 2}
    arguments.update(kwargs)
    result = import_fnspid(paths, tmp_path / "imported", **arguments)
    return pd.read_parquet(result["articles_path"]), result["report"], result


def test_deduplicate_across_csvs_preserve_tickers_and_raw_provenance(tmp_path):
    base = ["2020-01-05T12:00:00Z", " Profits\n  rise ", " aapl ",
            "https://example.com/a?utm_source=test#fragment", "Publisher", "DO NOT SCORE"]
    first = _csv(tmp_path, "first.csv", [base, base])
    second = _csv(tmp_path, "second.csv", [base, [*base[:2], "JPM", "https://EXAMPLE.com/a", *base[4:]]])
    frame, report, result = _import(tmp_path, [second, first])
    assert len(frame) == 2
    assert frame.news_id.nunique() == 1
    assert frame.text.tolist() == ["Profits rise", "Profits rise"]
    assert frame.ticker.tolist() == ["AAPL", "JPM"]
    assert frame.url.eq("https://example.com/a").all()
    assert frame.content_kind.eq("title").all()
    assert frame.association_kind.eq("dataset_stock_symbol_unverified").all()
    assert "DO NOT SCORE" not in frame.to_string()
    assert len(frame.loc[0, "source_files"]) == 2
    assert frame.loc[0, "raw_ticker"] == " aapl "
    assert report["counts"]["duplicate_associations"] == 2
    assert report["counts"]["articles_with_multiple_tickers"] == 1
    assert report["protocol"] == "fnspid_exploratory"
    assert report["historical_availability_proven"] is False
    assert report["source_revision"] == "a" * 40
    assert report["coverage"]["AAPL"]["counts_by_month"] == {"2020-01": 1, "2020-02": 0, "2020-03": 0}
    assert report["coverage"]["AAPL"]["months_without_rows"] == ["2020-02", "2020-03"]
    assert str(frame.published_at.dtype) == "datetime64[ns, UTC]"
    assert str(frame.collected_at.dtype) == "datetime64[ns, UTC]"
    assert "available_at" not in frame
    assert result["report_path"].is_file()


def test_retain_title_versions_and_flag_unproven_chronology(tmp_path):
    rows = [["2020-01-05T12:00:00Z", text, "AAPL", "https://example.com/a", "Publisher", "derived"]
            for text in ("Initial title", "Updated title")]
    rows.append(["2020-01-06T12:00:00Z", "Initial title", "JPM", "https://example.com/a", "Publisher", "derived"])
    frame, report, _ = _import(tmp_path, [_csv(tmp_path, "versions.csv", rows)])
    assert len(frame) == 3 and frame.news_id.nunique() == 2
    assert frame.version_timing_uncertain.all()
    assert frame.url_version_count.eq(2).all()
    assert frame.loc[frame.text.eq("Initial title"), "publication_timing_uncertain"].all()
    assert not frame.loc[frame.text.eq("Updated title"), "publication_timing_uncertain"].any()
    assert report["counts"]["urls_with_multiple_title_versions"] == 1
    assert report["counts"]["articles_with_publication_conflicts"] == 1


def test_filters_dates_exclusive_end_and_counts_rejected_rows(tmp_path):
    rows = [
        ["2020-01-01", "Accepted", "AAPL", "", "Publisher", ""],
        ["bad-date", "Bad date", "AAPL", "", "Publisher", ""],
        ["2020-02-01", " \t ", "AAPL", "", "Publisher", ""],
        ["bad-date", "", "AAPL", "", "Publisher", ""],
        ["2020-04-01", "Exclusive end", "AAPL", "", "Publisher", ""],
        ["2019-12-31", "Before start", "AAPL", "", "Publisher", ""],
        ["2020-02-01", "Other ticker", "XOM", "", "Publisher", ""],
    ]
    frame, report, _ = _import(tmp_path, [_csv(tmp_path, "filtered.csv", rows)])
    assert frame.text.tolist() == ["Accepted"]
    counts = report["counts"]
    assert counts["input_rows"] == 7
    assert counts["invalid_date_rows"] == counts["empty_title_rows"] == 2
    assert counts["rejected_rows"] == 3
    assert counts["filtered_period_rows"] == 2
    assert counts["filtered_ticker_rows"] == 1
    assert report["naive_date_rows_assumed_timezone"] == 1


def test_naive_timezone_is_explicit_and_aware_dates_keep_their_offset(tmp_path):
    rows = [[date, "Title " + str(index), "AAPL", "", "", ""]
            for index, date in enumerate(("2020-01-01 00:30:00", "2020-01-01T00:30:00-05:00"))]
    frame, report, _ = _import(tmp_path, [_csv(tmp_path, "dates.csv", rows)],
                               start="2019-12-31T00:00:00Z", end="2020-01-02T00:00:00Z",
                               timezone_assumption="America/New_York")
    assert frame.published_at.eq(pd.Timestamp("2020-01-01T05:30:00Z")).all()
    assert report["settings"]["timezone_assumption"] == "America/New_York"
    assert report["naive_date_rows_assumed_timezone"] == 1


def test_dst_ambiguous_and_nonexistent_times_are_rejected(tmp_path):
    rows = [[date, "Title", "AAPL", "", "", ""]
            for date in ("2020-03-08 02:30:00", "2020-11-01 01:30:00")]
    frame, report, _ = _import(tmp_path, [_csv(tmp_path, "dates.csv", rows)],
                               start="2020-01-01", end="2021-01-01", timezone_assumption="America/New_York")
    assert frame.empty
    assert report["counts"]["invalid_date_rows"] == 2


@pytest.mark.parametrize("row", [
    ["2020-01-01", "Title", "AAPL", "url", "source", "derived", "EXTRA"],
    ["2020-01-01", "Title", "AAPL"],
    ["2020-01-01", "Title", "XOM", "url", "source", "derived", "EXTRA"],
])
def test_malformed_rows_fail_even_when_ticker_would_be_filtered(tmp_path, row):
    path = _csv(tmp_path, "malformed.csv", [row])
    with pytest.raises(ValueError, match="Malformed CSV row"):
        _import(tmp_path, [path])
    assert not (tmp_path / "imported" / "articles.parquet").exists()


def test_malformed_quoted_record_fails(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text(
        "Date,Article_title,Stock_symbol\n2020-01-01,\"unterminated,AAPL\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed FNSPID CSV"):
        _import(tmp_path, [path])


@pytest.mark.parametrize("revision", ["main", "f" * 39, "A" * 40, None])
def test_revision_must_be_immutable(tmp_path, revision):
    path = _csv(tmp_path, "data.csv", [["2020-01-01", "Title", "AAPL", "", "", ""]])
    with pytest.raises(ValueError, match="immutable"):
        _import(tmp_path, [path], dataset_info=_info(tmp_path, revision))


@pytest.mark.parametrize("header", [["Date", "Stock_symbol"], ["Date", "Article_title", "Stock_symbol", "Date"]])
def test_missing_and_duplicate_headers_fail(tmp_path, header):
    with pytest.raises(ValueError, match="columns"):
        _import(tmp_path, [_csv(tmp_path, "schema.csv", [], header)])


def test_resume_validates_unchanged_inputs_and_corrupted_artifact(tmp_path):
    path = _csv(tmp_path, "data.csv", [["2020-01-01", "Title", "AAPL", "", "", ""]])
    original, report, result = _import(tmp_path, [path])
    resumed, new_report, _ = _import(tmp_path, [path], resume=True)
    pd.testing.assert_frame_equal(original, resumed)
    assert report == new_report
    with pytest.raises(FileExistsError):
        _import(tmp_path, [path])
    result["articles_path"].write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="corrupted"):
        _import(tmp_path, [path], resume=True)


def test_resume_refuses_changed_source_and_settings(tmp_path):
    path = _csv(tmp_path, "data.csv", [["2020-01-01", "Title", "AAPL", "", "", ""]])
    _import(tmp_path, [path])
    with pytest.raises(ValueError, match="settings changed"):
        _import(tmp_path, [path], resume=True, tickers=["AAPL"])
    _csv(tmp_path, "data.csv", [["2020-01-01", "Changed", "AAPL", "", "", ""]])
    with pytest.raises(ValueError, match="input_files changed"):
        _import(tmp_path, [path], resume=True)


def test_empty_filtered_output_retains_schema_and_reports_gaps(tmp_path):
    path = _csv(tmp_path, "data.csv", [["2020-01-01", "Title", "XOM", "", "", ""]])
    frame, report, _ = _import(tmp_path, [path])
    assert frame.empty
    assert str(frame.published_at.dtype) == "datetime64[ns, UTC]"
    assert report["coverage"]["AAPL"]["months_without_rows"] == ["2020-01", "2020-02", "2020-03"]
    assert report["coverage"]["AAPL"]["first"] is None


def test_concurrent_import_refuses_the_locked_destination(tmp_path):
    path = _csv(tmp_path, "data.csv", [["2020-01-01", "Title", "AAPL", "", "", ""]])
    output = tmp_path / "imported"
    output.mkdir()
    with _output_lock(output), pytest.raises(ValueError, match="already in use"):
        _import(tmp_path, [path])
