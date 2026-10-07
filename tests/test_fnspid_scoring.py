"""Small offline fixtures for the resumable FNSPID FinBERT adapter."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from trading_system.data import fnspid_scoring
from trading_system.data.fnspid_scoring import score_fnspid
from trading_system.data.news_collection import _output_lock
from trading_system.data.news_pilot_scoring import FinBERTCheckpoint, file_sha256


CHECKPOINT = FinBERTCheckpoint("yiyanghkust/finbert-tone", "a" * 40, "b" * 64)


class Analyzer:
    def __init__(self, fail_call=None):
        self.calls = []
        self.fail_call = fail_call

    def predict(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_call:
            raise RuntimeError("deliberate offline inference failure")
        return pd.DataFrame({
            "text": texts, "label": "positive", "p_negative": 0.1,
            "p_neutral": 0.1, "p_positive": 0.8, "confidence": 0.8,
            "sentiment_score": 0.7,
        })


@pytest.fixture(autouse=True)
def offline_device(monkeypatch):
    # The scoring contract tests do not need to initialize a real GPU runtime.
    monkeypatch.setattr(fnspid_scoring, "_resolved_device", lambda device, **kwargs: "cpu")


def _articles(tmp_path, texts=("Profits rise", "Profits rise", "Rates fall")):
    path = tmp_path / "articles.parquet"
    frame = pd.DataFrame({
        "news_id": [f"article-{index}" for index in range(len(texts))],
        "ticker": ["AAPL" if index % 2 == 0 else "JPM" for index in range(len(texts))],
        "text_hash": [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts],
        "text": texts, "source": "fixture", "url": "https://example.com/news",
        "published_at": pd.Timestamp("2023-05-01T10:00:00Z"),
        "collected_at": pd.Timestamp("2026-10-06T10:00:00Z"),
        "date_raw": "2023-05-01T10:00:00Z", "association_kind": "dataset_symbol",
        "content_kind": "title", "source_files": [["Stock_news/All_external.csv"]] * len(texts),
        "version_timing_uncertain": False, "publication_timing_uncertain": False,
    })
    frame.to_parquet(path, index=False)
    return path


def _run(tmp_path, source=None, analyzer=None, **kwargs):
    source = source or _articles(tmp_path)
    analyzer = analyzer or Analyzer()
    result = score_fnspid(source, tmp_path / "scored", checkpoint=CHECKPOINT,
                          analyzer_factory=lambda: analyzer, device="cpu", **kwargs)
    return source, analyzer, result


def test_unique_texts_are_scored_once_then_routed_to_all_associations(tmp_path):
    source, analyzer, result = _run(tmp_path, batch_size=1, shard_size=1)
    original = pd.read_parquet(source)
    scored = pd.read_parquet(result["scored_path"])
    expected = original[["text_hash", "text"]].drop_duplicates().sort_values("text_hash")["text"].tolist()
    assert [text for call in analyzer.calls for text in call] == expected
    assert scored["ticker"].tolist() == original["ticker"].tolist()
    assert scored["news_id"].tolist() == original["news_id"].tolist()
    pd.testing.assert_frame_equal(scored[list(original)], original)
    assert scored["sentiment_score"].eq(0.7).all()
    assert scored["checkpoint"].eq(CHECKPOINT.identifier).all()
    manifest = result["manifest"]
    assert manifest["state"] == "complete" and manifest["unique_texts"] == 2
    assert len(manifest["shards"]) == 2
    assert manifest["input_identifiers"]["articles_sha256"] == file_sha256(source)
    assert set(manifest["runtime_versions"]) == {"torch", "transformers", "news-sentiment-feature-engineering"}


def test_same_news_id_can_route_to_different_tickers(tmp_path):
    source = _articles(tmp_path, texts=("Profits rise", "Profits rise"))
    frame = pd.read_parquet(source)
    frame["news_id"] = "same-news"
    frame.to_parquet(source, index=False)
    _, analyzer, result = _run(tmp_path, source)
    assert analyzer.calls == [["Profits rise"]]
    assert pd.read_parquet(result["scored_path"])["ticker"].tolist() == ["AAPL", "JPM"]


def test_resume_completed_does_not_load_or_call_analyzer(tmp_path):
    source, _, result = _run(tmp_path)
    manifest_bytes = result["manifest_path"].read_bytes()
    resumed = score_fnspid(source, tmp_path / "scored", checkpoint=CHECKPOINT, device="cpu", resume=True)
    assert resumed["manifest_path"].read_bytes() == manifest_bytes
    assert resumed["scored_path"] == result["scored_path"]


def test_resume_after_inference_failure_reuses_completed_shards(tmp_path):
    source = _articles(tmp_path, texts=tuple(f"News {index}" for index in range(5)))
    failed = Analyzer(fail_call=3)
    with pytest.raises(RuntimeError, match="deliberate"):
        _run(tmp_path, source, failed, batch_size=1, shard_size=2)
    output = tmp_path / "scored"
    manifest = json.loads((output / "scoring.manifest.json").read_text())
    assert manifest["state"] == "running" and len(manifest["shards"]) == 1
    first_shard = output / manifest["shards"][0]["path"]
    original_shard = first_shard.read_bytes()
    resumed = Analyzer()
    result = score_fnspid(source, output, checkpoint=CHECKPOINT, device="cpu",
                          analyzer_factory=lambda: resumed, batch_size=1, shard_size=2, resume=True)
    expected_texts = pd.read_parquet(source).sort_values("text_hash")["text"].tolist()[2:]
    assert [text for call in resumed.calls for text in call] == expected_texts
    assert first_shard.read_bytes() == original_shard
    assert result["manifest"]["state"] == "complete"


@pytest.mark.parametrize("mutation", ["input", "checkpoint", "batch_size", "max_length", "shard_size", "runtime"])
def test_resume_rejects_changed_signature_before_inference(tmp_path, mutation, monkeypatch):
    source, _, result = _run(tmp_path)
    checkpoint = CHECKPOINT
    extra = {}
    if mutation == "input":
        frame = pd.read_parquet(source)
        frame["url"] = "https://example.com/changed"
        frame.to_parquet(source, index=False)
    elif mutation == "checkpoint":
        checkpoint = FinBERTCheckpoint(CHECKPOINT.repository, "c" * 40, CHECKPOINT.weights_sha256)
    elif mutation == "runtime":
        monkeypatch.setattr(fnspid_scoring, "_versions", lambda: {"torch": "changed"})
    else:
        extra[mutation] = 2
    analyzer = Analyzer()
    original_manifest = result["manifest_path"].read_bytes()
    with pytest.raises(ValueError, match="changed input, checkpoint"):
        score_fnspid(source, tmp_path / "scored", checkpoint=checkpoint, device="cpu",
                      analyzer_factory=lambda: analyzer, resume=True, **extra)
    assert not analyzer.calls
    assert result["manifest_path"].read_bytes() == original_manifest


@pytest.mark.parametrize("artifact", ["shard", "company"])
def test_resume_rejects_corrupted_artifacts_without_overwriting(tmp_path, artifact):
    source, _, result = _run(tmp_path)
    target = result["scored_path"] if artifact == "company" else tmp_path / "scored" / result["manifest"]["shards"][0]["path"]
    frame = pd.read_parquet(target)
    frame["confidence"] = 0.4
    frame.to_parquet(target, index=False)
    changed_bytes = target.read_bytes()
    analyzer = Analyzer()
    with pytest.raises(ValueError, match="checksum"):
        score_fnspid(source, tmp_path / "scored", checkpoint=CHECKPOINT, device="cpu",
                      analyzer_factory=lambda: analyzer, resume=True)
    assert not analyzer.calls and target.read_bytes() == changed_bytes


def test_resume_refuses_unregistered_orphan_shard(tmp_path):
    source, _, result = _run(tmp_path)
    orphan = tmp_path / "scored" / "shards" / "part-999999.parquet"
    pd.read_parquet(result["scored_path"]).to_parquet(orphan, index=False)
    with pytest.raises(ValueError, match="unregistered"):
        score_fnspid(source, tmp_path / "scored", checkpoint=CHECKPOINT, device="cpu", resume=True)
    assert orphan.exists()


def test_rehashed_corrupted_shard_still_fails_prediction_validation(tmp_path):
    source, _, result = _run(tmp_path)
    entry = result["manifest"]["shards"][0]
    target = tmp_path / "scored" / entry["path"]
    frame = pd.read_parquet(target)
    frame["sentiment_score"] = -0.4
    frame.to_parquet(target, index=False)
    manifest = json.loads(result["manifest_path"].read_text())
    manifest["shards"][0]["sha256"] = file_sha256(target)
    result["manifest_path"].write_text(json.dumps(manifest))
    analyzer = Analyzer()
    with pytest.raises(ValueError, match="invalid shard"):
        score_fnspid(source, tmp_path / "scored", checkpoint=CHECKPOINT, device="cpu",
                      analyzer_factory=lambda: analyzer, resume=True)
    assert not analyzer.calls


def test_concurrent_artifact_creation_is_never_overwritten(tmp_path, monkeypatch):
    original_link = fnspid_scoring.os.link

    def concurrent_link(source, target):
        Path(target).write_bytes(b"concurrent artifact must survive")
        original_link(source, target)

    monkeypatch.setattr(fnspid_scoring.os, "link", concurrent_link)
    with pytest.raises(FileExistsError):
        _run(tmp_path)
    target = tmp_path / "scored" / "shards" / "part-000000.parquet"
    assert target.read_bytes() == b"concurrent artifact must survive"


def test_manifest_window_after_shard_commit_is_explicitly_refused(tmp_path, monkeypatch):
    source = _articles(tmp_path)
    original_writer = fnspid_scoring._write_json
    writes = []

    def fail_after_prepared(path, payload):
        writes.append(path)
        if len(writes) == 3:
            raise OSError("deliberate manifest commit failure")
        original_writer(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(fnspid_scoring, "_write_json", fail_after_prepared)
        with pytest.raises(OSError, match="manifest commit"):
            _run(tmp_path, source, shard_size=1)
    output = tmp_path / "scored"
    manifest = json.loads((output / "scoring.manifest.json").read_text())
    assert manifest["shards"][0]["status"] == "prepared"
    target = output / manifest["shards"][0]["path"]
    original_bytes = target.read_bytes()
    with pytest.raises(ValueError, match="interrupted prepared artifact commit"):
        score_fnspid(source, output, checkpoint=CHECKPOINT, device="cpu", shard_size=1, resume=True)
    assert target.read_bytes() == original_bytes


@pytest.mark.parametrize("column,value", [
    ("p_positive", 0.7), ("p_negative", float("nan")), ("p_negative", -0.1),
    ("confidence", 0.4), ("sentiment_score", -0.3), ("label", "negative"),
])
def test_invalid_predictions_never_commit_a_shard(tmp_path, column, value):
    def invalid(texts):
        frame = Analyzer().predict(texts)
        frame[column] = value
        return frame
    with pytest.raises(ValueError):
        _run(tmp_path, analyzer=invalid)
    manifest = json.loads((tmp_path / "scored" / "scoring.manifest.json").read_text())
    assert not manifest["shards"] and not manifest["artifacts"]


def test_prediction_text_order_is_checked(tmp_path):
    def reordered(texts):
        return Analyzer().predict(texts[::-1])
    with pytest.raises(ValueError, match="text/order/count"):
        _run(tmp_path, analyzer=reordered)


def test_local_model_metadata_and_weights_are_verified_for_resume(tmp_path):
    source = _articles(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"architectures": ["BertForSequenceClassification"]}')
    (model / "vocab.txt").write_text("[PAD]\n[UNK]\n")
    weights = model / "pytorch_model.bin"
    weights.write_bytes(b"small offline fixture, not actual model weights")
    checkpoint = FinBERTCheckpoint(CHECKPOINT.repository, CHECKPOINT.revision, file_sha256(weights))
    result = score_fnspid(source, tmp_path / "scored", checkpoint=checkpoint, model_dir=model,
                          analyzer_factory=Analyzer, device="cpu")
    metadata = result["manifest"]["model"]["metadata_sha256"]
    assert metadata["config.json"] == file_sha256(model / "config.json")
    assert metadata["vocab.txt"] == file_sha256(model / "vocab.txt")
    (model / "vocab.txt").write_text("[PAD]\n[UNK]\nnew token\n")
    with pytest.raises(ValueError, match="changed input, checkpoint"):
        score_fnspid(source, tmp_path / "scored", checkpoint=checkpoint, model_dir=model, device="cpu", resume=True)
    weights.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="weights do not match"):
        score_fnspid(source, tmp_path / "scored", checkpoint=checkpoint, model_dir=model, device="cpu", resume=True)


def test_timing_uncertain_associations_are_excluded_with_audit_counts(tmp_path):
    source = _articles(tmp_path)
    frame = pd.read_parquet(source)
    frame["version_timing_uncertain"] = [False, True, True]
    frame["publication_timing_uncertain"] = [False, False, True]
    frame.to_parquet(source, index=False)
    _, analyzer, result = _run(tmp_path, source)
    assert analyzer.calls == [["Profits rise"]]
    assert pd.read_parquet(result["scored_path"])["news_id"].tolist() == ["article-0"]
    assert result["manifest"]["timing_rejections"] == {
        "version_timing_uncertain": 2, "publication_timing_uncertain": 1, "excluded_associations": 2,
    }


def test_all_timing_uncertain_news_export_empty_schema_without_inference(tmp_path):
    source = _articles(tmp_path)
    frame = pd.read_parquet(source)
    frame["publication_timing_uncertain"] = True
    frame.to_parquet(source, index=False)
    _, analyzer, result = _run(tmp_path, source)
    assert not analyzer.calls and not result["manifest"]["shards"]
    scored = pd.read_parquet(result["scored_path"])
    assert scored.empty and {"p_positive", "confidence", "ticker", "checkpoint"}.issubset(scored)


def test_missing_timing_flags_are_accepted_as_false_for_adapters(tmp_path):
    source = _articles(tmp_path)
    frame = pd.read_parquet(source).drop(columns=list(fnspid_scoring.TIMING_FLAGS))
    frame.to_parquet(source, index=False)
    _, _, result = _run(tmp_path, source)
    assert result["manifest"]["timing_rejections"]["excluded_associations"] == 0


def test_invalid_timing_flag_types_are_rejected(tmp_path):
    source = _articles(tmp_path)
    frame = pd.read_parquet(source)
    frame["publication_timing_uncertain"] = 1
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError, match="non-null boolean"):
        _run(tmp_path, source)
    assert not (tmp_path / "scored").exists()


@pytest.mark.parametrize("mutation", ["hash", "association", "identity"])
def test_input_identity_and_hashes_are_checked_before_creating_output(tmp_path, mutation):
    source = _articles(tmp_path)
    frame = pd.read_parquet(source)
    if mutation == "hash":
        frame.loc[0, "text_hash"] = "d" * 64
    elif mutation == "association":
        frame.loc[1, "news_id"] = frame.loc[0, "news_id"]
        frame.loc[1, "ticker"] = frame.loc[0, "ticker"]
    else:
        frame.loc[2, "news_id"] = frame.loc[0, "news_id"]
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError):
        _run(tmp_path, source)
    assert not (tmp_path / "scored").exists()


def test_existing_version_is_never_overwritten_without_resume(tmp_path):
    source, _, result = _run(tmp_path)
    original_bytes = result["scored_path"].read_bytes()
    with pytest.raises(FileExistsError):
        _run(tmp_path, source)
    assert result["scored_path"].read_bytes() == original_bytes


def test_existing_nonempty_directory_without_manifest_is_refused(tmp_path):
    source = _articles(tmp_path)
    output = tmp_path / "scored"
    output.mkdir()
    existing = output / "unrelated.parquet"
    pd.read_parquet(source).to_parquet(existing, index=False)
    with pytest.raises(ValueError, match="missing or invalid manifest"):
        score_fnspid(source, output, checkpoint=CHECKPOINT, device="cpu", resume=True)
    assert existing.exists()


def test_parallel_writer_is_rejected_before_inference(tmp_path):
    source = _articles(tmp_path)
    output = tmp_path / "scored"
    output.mkdir()
    analyzer = Analyzer()
    with _output_lock(output):
        with pytest.raises(ValueError, match="already in use"):
            score_fnspid(source, output, checkpoint=CHECKPOINT, device="cpu", analyzer_factory=lambda: analyzer)
    assert not analyzer.calls and not (output / "scoring.manifest.json").exists()


def _reuse_cache(tmp_path, name="pilot", texts=("Profits rise", "Rates fall"), **kwargs):
    directory = tmp_path / name
    directory.mkdir()
    articles = _articles(directory, texts=texts)
    result = score_fnspid(articles, directory / "scored", checkpoint=CHECKPOINT,
                          analyzer_factory=Analyzer, device="cpu", shard_size=1, **kwargs)
    return result


@pytest.mark.parametrize("cache_lock_present", [True, False])
def test_completed_cache_reuses_all_texts_without_loading_analyzer(tmp_path, cache_lock_present):
    cache = _reuse_cache(tmp_path)
    cache_dir = cache["scored_path"].parent
    cache_lock = cache_dir / ".collection.lock"
    if not cache_lock_present:
        cache_lock.unlink()
    original_files = {
        path.relative_to(cache_dir).as_posix(): file_sha256(path)
        for path in cache_dir.rglob("*") if path.is_file()
    }
    source = _articles(tmp_path)

    def forbidden():
        pytest.fail("Cached texts must not load an analyzer")

    result = score_fnspid(source, tmp_path / "extended", checkpoint=CHECKPOINT,
                          device="auto", shard_size=2, analyzer_factory=forbidden,
                          reuse_from=[cache_dir])
    manifest = result["manifest"]
    assert manifest["reused_unique_texts"] == 2
    assert manifest["scored_new_unique_texts"] == 0
    assert manifest["reuse_sources"][0]["manifest_sha256"] == file_sha256(cache["manifest_path"])
    assert manifest["signature"]["reuse_sources"] == manifest["reuse_sources"]
    assert pd.read_parquet(result["scored_path"])["sentiment_score"].eq(0.7).all()
    assert len(manifest["shards"]) == 1
    assert cache_lock.exists() is cache_lock_present
    assert {
        path.relative_to(cache_dir).as_posix(): file_sha256(path)
        for path in cache_dir.rglob("*") if path.is_file()
    } == original_files


def test_all_cached_texts_work_without_analyzer_or_model_directory(tmp_path):
    cache = _reuse_cache(tmp_path)
    source = _articles(tmp_path)
    result = score_fnspid(source, tmp_path / "extended", checkpoint=CHECKPOINT,
                          device="cpu", reuse_from=[cache["scored_path"].parent])
    assert result["manifest"]["state"] == "complete"
    original = result["manifest_path"].read_bytes()
    resumed = score_fnspid(source, tmp_path / "extended", checkpoint=CHECKPOINT,
                           device="cpu", reuse_from=[cache["scored_path"].parent], resume=True)
    assert resumed["manifest_path"].read_bytes() == original


def test_cache_only_infers_missing_unique_texts_and_routes_mixed_shards(tmp_path):
    cache = _reuse_cache(tmp_path)
    source = _articles(tmp_path, texts=("New headline", "Profits rise", "Rates fall", "New headline"))
    analyzer = Analyzer()
    result = score_fnspid(source, tmp_path / "extended", checkpoint=CHECKPOINT,
                          device="cpu", analyzer_factory=lambda: analyzer,
                          shard_size=3, reuse_from=[cache["scored_path"].parent])
    assert analyzer.calls == [["New headline"]]
    assert result["manifest"]["reused_unique_texts"] == 2
    assert result["manifest"]["scored_new_unique_texts"] == 1
    scored = pd.read_parquet(result["scored_path"])
    pd.testing.assert_frame_equal(scored[list(pd.read_parquet(source))], pd.read_parquet(source))
    assert result["manifest"]["shards"][0]["start"] == 0
    assert result["manifest"]["shards"][0]["end"] == 3


@pytest.mark.parametrize("artifact", [
    "shard", "company", "missing_shard", "missing_company", "orphan",
    "nested_lock", "lock_directory", "incomplete",
])
def test_reuse_rejects_corrupt_external_cache_before_creating_target(tmp_path, artifact):
    cache = _reuse_cache(tmp_path)
    cache_dir = cache["scored_path"].parent
    if artifact == "shard":
        (cache_dir / cache["manifest"]["shards"][0]["path"]).write_bytes(b"corrupt cache")
    elif artifact == "company":
        cache["scored_path"].write_bytes(b"corrupt company route")
    elif artifact == "missing_shard":
        (cache_dir / cache["manifest"]["shards"][0]["path"]).unlink()
    elif artifact == "missing_company":
        cache["scored_path"].unlink()
    elif artifact == "orphan":
        (cache_dir / "orphan.parquet").write_bytes(b"unregistered cache")
    elif artifact == "nested_lock":
        (cache_dir / "shards" / ".collection.lock").write_bytes(b"unregistered lock")
    elif artifact == "lock_directory":
        lock = cache_dir / ".collection.lock"
        lock.unlink()
        lock.mkdir()
    else:
        manifest = cache["manifest"]
        manifest["state"] = "running"
        cache["manifest_path"].write_text(json.dumps(manifest))
    source = _articles(tmp_path)
    target = tmp_path / "extended"
    with pytest.raises(ValueError, match="reuse refused"):
        score_fnspid(source, target, checkpoint=CHECKPOINT, device="cpu",
                      analyzer_factory=lambda: pytest.fail("Must validate cache before inference"),
                      reuse_from=[cache_dir])
    assert not target.exists()


@pytest.mark.parametrize("drift", ["checkpoint", "batch_size", "max_length", "resolved_device", "runtime", "library"])
def test_reuse_rejects_incompatible_inference_identity_before_writes(tmp_path, drift, monkeypatch):
    cache = _reuse_cache(tmp_path)
    source = _articles(tmp_path)
    extra = {}
    checkpoint = CHECKPOINT
    if drift == "checkpoint":
        checkpoint = FinBERTCheckpoint(CHECKPOINT.repository, "c" * 40, CHECKPOINT.weights_sha256)
    elif drift == "resolved_device":
        monkeypatch.setattr(fnspid_scoring, "_resolved_device", lambda device, **kwargs: "cuda")
    elif drift == "runtime":
        monkeypatch.setattr(fnspid_scoring, "_versions", lambda: {"torch": "new-runtime"})
    elif drift == "library":
        monkeypatch.setattr(fnspid_scoring, "LIBRARY_REVISION", "new-revision")
    else:
        extra[drift] = 2
    target = tmp_path / "extended"
    with pytest.raises(ValueError, match="incompatible cache"):
        score_fnspid(source, target, checkpoint=checkpoint, device="cpu",
                      analyzer_factory=lambda: pytest.fail("Must reject inference drift"),
                      reuse_from=[cache["scored_path"].parent], **extra)
    assert not target.exists()


def test_reuse_rejects_changed_model_tokenization_metadata(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "vocab.txt").write_text("[PAD]\n[UNK]\n")
    weights = model / "pytorch_model.bin"
    weights.write_bytes(b"offline test weights")
    checkpoint = FinBERTCheckpoint(CHECKPOINT.repository, CHECKPOINT.revision, file_sha256(weights))
    source = _articles(tmp_path)
    cache = score_fnspid(source, tmp_path / "cache", checkpoint=checkpoint, model_dir=model,
                         analyzer_factory=Analyzer, device="cpu")
    (model / "vocab.txt").write_text("[PAD]\n[UNK]\nchanged-token\n")
    with pytest.raises(ValueError, match="incompatible cache model_metadata"):
        score_fnspid(source, tmp_path / "extended", checkpoint=checkpoint, model_dir=model,
                      analyzer_factory=Analyzer, device="cpu", reuse_from=[cache["scored_path"].parent])
    assert not (tmp_path / "extended").exists()


def test_reuse_validates_text_hash_provenance_even_after_rehashing_cache(tmp_path):
    cache = _reuse_cache(tmp_path)
    source = _articles(tmp_path)
    company = pd.read_parquet(cache["scored_path"])
    company.loc[0, "text"] = "Changed cached text"
    company.to_parquet(cache["scored_path"], index=False)
    manifest = cache["manifest"]
    manifest["artifacts"]["scored_company.parquet"]["sha256"] = file_sha256(cache["scored_path"])
    cache["manifest_path"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="text_hash does not match"):
        score_fnspid(source, tmp_path / "extended", checkpoint=CHECKPOINT,
                      device="cpu", reuse_from=[cache["scored_path"].parent])
    assert not (tmp_path / "extended").exists()


def test_resume_rejects_changed_external_cache_manifest_without_writes(tmp_path):
    cache = _reuse_cache(tmp_path)
    source = _articles(tmp_path)
    result = score_fnspid(source, tmp_path / "extended", checkpoint=CHECKPOINT,
                          device="cpu", reuse_from=[cache["scored_path"].parent])
    original_target = result["manifest_path"].read_bytes()
    # Even an otherwise valid source whose manifest bytes changed is a different
    # cache provenance. Existing target artifacts must retain their identity.
    manifest = cache["manifest"]
    manifest["coverage_note"] = "Changed source metadata"
    cache["manifest_path"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed input, checkpoint"):
        score_fnspid(source, tmp_path / "extended", checkpoint=CHECKPOINT, device="cpu",
                      reuse_from=[cache["scored_path"].parent], resume=True)
    assert result["manifest_path"].read_bytes() == original_target


def test_reuse_rejects_different_valid_scores_for_same_text_across_sources(tmp_path):
    first = _reuse_cache(tmp_path, name="first")
    second = _reuse_cache(tmp_path, name="second")
    manifest = second["manifest"]
    paths = [second["scored_path"], *[
        second["scored_path"].parent / entry["path"] for entry in manifest["shards"]
    ]]
    for path in paths:
        frame = pd.read_parquet(path)
        frame["p_negative"] = 0.2
        frame["p_positive"] = 0.7
        frame["confidence"] = 0.7
        frame["sentiment_score"] = 0.5
        frame.to_parquet(path, index=False)
    manifest["artifacts"]["scored_company.parquet"]["sha256"] = file_sha256(second["scored_path"])
    for entry in manifest["shards"]:
        entry["sha256"] = file_sha256(second["scored_path"].parent / entry["path"])
    second["manifest_path"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="conflicting cached text or scores"):
        score_fnspid(_articles(tmp_path), tmp_path / "extended", checkpoint=CHECKPOINT,
                      device="cpu", reuse_from=[first["scored_path"].parent, second["scored_path"].parent])
    assert not (tmp_path / "extended").exists()


def test_mixed_cache_resume_keeps_committed_shards_and_only_scores_remaining_misses(tmp_path):
    texts = tuple(f"Headline {index}" for index in range(6))
    sorted_texts = sorted(texts, key=lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    cache = _reuse_cache(tmp_path, texts=(sorted_texts[0], sorted_texts[2]))
    source = _articles(tmp_path, texts=texts)
    output = tmp_path / "extended"
    failing = Analyzer(fail_call=2)
    with pytest.raises(RuntimeError, match="deliberate"):
        score_fnspid(source, output, checkpoint=CHECKPOINT, device="cpu", shard_size=2,
                      analyzer_factory=lambda: failing, reuse_from=[cache["scored_path"].parent])
    first_shard = output / "shards/part-000000.parquet"
    original = first_shard.read_bytes()
    analyzer = Analyzer()
    result = score_fnspid(source, output, checkpoint=CHECKPOINT, device="cpu", shard_size=2,
                          analyzer_factory=lambda: analyzer, reuse_from=[cache["scored_path"].parent], resume=True)
    assert [text for call in analyzer.calls for text in call] == sorted_texts[3:]
    assert first_shard.read_bytes() == original
    assert result["manifest"]["state"] == "complete"
    assert result["manifest"]["reused_unique_texts"] == 2
    assert result["manifest"]["scored_new_unique_texts"] == 4


def test_multiple_reuse_sources_have_deterministic_deduplicated_provenance(tmp_path):
    first = _reuse_cache(tmp_path, name="first", texts=("Profits rise",))
    second = _reuse_cache(tmp_path, name="second", texts=("Profits rise", "Rates fall"))
    source = _articles(tmp_path)
    directories = [second["scored_path"].parent, first["scored_path"].parent]
    result = score_fnspid(source, tmp_path / "extended", checkpoint=CHECKPOINT, device="cpu",
                          reuse_from=[*directories, directories[0]])
    assert len(result["manifest"]["reuse_sources"]) == 2
    assert result["manifest"]["reused_unique_texts"] == 2
    original_manifest = result["manifest_path"].read_bytes()
    score_fnspid(source, tmp_path / "extended", checkpoint=CHECKPOINT, device="cpu",
                  reuse_from=directories[::-1], resume=True)
    assert result["manifest_path"].read_bytes() == original_manifest


def test_reuse_rejects_rehashed_company_scores_that_disagree_with_shards(tmp_path):
    cache = _reuse_cache(tmp_path)
    company = pd.read_parquet(cache["scored_path"])
    company["p_negative"] = 0.2
    company["p_positive"] = 0.7
    company["confidence"] = 0.7
    company["sentiment_score"] = 0.5
    company.to_parquet(cache["scored_path"], index=False)
    cache["manifest"]["artifacts"]["scored_company.parquet"]["sha256"] = file_sha256(cache["scored_path"])
    cache["manifest_path"].write_text(json.dumps(cache["manifest"]))
    with pytest.raises(ValueError, match="reuse refused"):
        score_fnspid(_articles(tmp_path), tmp_path / "extended", checkpoint=CHECKPOINT, device="cpu",
                      reuse_from=[cache["scored_path"].parent])
    assert not (tmp_path / "extended").exists()

