"""Private study reuse keeps verified TRAIN preparation and sealed OUTER data."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from test_fnspid_polarity_study import _inputs
from test_fnspid_protocol import _export
from trading_system.data import fnspid_polarity
from trading_system.experiments import news_sentiment_ablation as news


pytest.importorskip("torch")


def _plan(inputs, *, cache=None, runtime=None, **changes):
    market, exported, config, loss, parameters, ablation, _, _ = inputs
    options = dict(frame=market, config=config, loss=loss, gru_parameters=parameters,
                   seeds=[1], sentiment_export=exported, n_splits=2, gap_bars=2,
                   ablation=ablation, _study_cache=cache, _runtime_cache=runtime)
    options.update(changes)
    return news.plan_run_news_sentiment_ablation(**options)


def _spies(monkeypatch):
    counts = dict(prepare=0, articles=0, article_hash=0, prefix=0, scaler=0)
    original_prepare = news._prepare
    original_articles = news._scored_articles
    original_hash = news._sha256
    original_control = fnspid_polarity.prepare_fold_news_control
    original_fit = news._SentimentScaler.fit

    def prepare(*args, **kwargs):
        counts["prepare"] += 1
        return original_prepare(*args, **kwargs)

    def articles(*args, **kwargs):
        counts["articles"] += 1
        return original_articles(*args, **kwargs)

    def digest(*args, **kwargs):
        counts["article_hash"] += 1
        return original_hash(*args, **kwargs)

    def control(*args, **kwargs):
        assert kwargs["include_outer"] is False, "The planner must not construct OUTER controls."
        counts["prefix"] += 1
        return original_control(*args, **kwargs)

    def fit(cls, *args, **kwargs):
        counts["scaler"] += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(news, "_prepare", prepare)
    monkeypatch.setattr(news, "_scored_articles", articles)
    monkeypatch.setattr(news, "_sha256", digest)
    monkeypatch.setattr(fnspid_polarity, "prepare_fold_news_control", control)
    monkeypatch.setattr(news._SentimentScaler, "fit", classmethod(fit))
    return counts


def test_cached_and_uncached_plans_have_identical_signatures_and_values(tmp_path):
    inputs = _inputs(tmp_path)
    expected = _plan(inputs)
    cache, first_runtime, second_runtime = {}, {}, {}
    cached = _plan(inputs, cache=cache, runtime=first_runtime)
    again = _plan(inputs, cache=cache, runtime=second_runtime)
    assert cached == expected == again
    for fold in (0, 1):
        first = first_runtime["prepared_folds"][fold]
        second = second_runtime["prepared_folds"][fold]
        assert first["prepared"] is second["prepared"]
        for mode in ("original", "shuffled", "neutralized"):
            assert first["exports"][mode] is second["exports"][mode]
            assert first["scalers"][mode] is second["scalers"][mode]


def test_price_and_article_reuse_cross_candidates_and_corpus_seeds(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    counts = _spies(monkeypatch)
    cache = {}
    _plan(inputs, cache=cache)
    _plan(inputs, cache=cache)
    assert counts == dict(prepare=2, articles=1, article_hash=2, prefix=6, scaler=6)
    changed = replace(inputs[5], shuffle_seed=271828)
    _plan(inputs, cache=cache, ablation=changed)
    # Only the two shuffled fold prefixes/scalers depend on the corpus seed.
    assert counts == dict(prepare=2, articles=1, article_hash=3, prefix=8, scaler=8)
    subset = replace(changed, candidates=("gru_features_shuffled",))
    _plan(inputs, cache=cache, ablation=subset, seeds=[7, 19])
    assert counts == dict(prepare=2, articles=1, article_hash=4, prefix=8, scaler=8)


@pytest.mark.parametrize("change", ["config", "dataset", "cv", "model"])
def test_effective_metadata_changes_invalidate_price_preparation(tmp_path, monkeypatch, change):
    inputs = _inputs(tmp_path)
    counts = _spies(monkeypatch)
    cache = {}
    first = _plan(inputs, cache=cache)
    if change == "config":
        changes = {"config": replace(inputs[2], context_len=6)}
    elif change == "dataset":
        market = inputs[0].copy()
        market["volume"] += 1
        changes = {"frame": market}
    elif change == "cv":
        changes = {"gap_bars": 3}
    else:
        changes = {"gru_parameters": {**inputs[4], "hidden_size": 5}}
    second = _plan(inputs, cache=cache, **changes)
    assert counts["prepare"] == 4
    assert counts["articles"] == 1
    assert counts["prefix"] == counts["scaler"] == 12
    assert first["task_specs"][0]["signature"] != second["task_specs"][0]["signature"]


def test_new_verified_scored_source_reloads_articles_and_news_only(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    counts = _spies(monkeypatch)
    cache = {}
    _plan(inputs, cache=cache)
    different = tmp_path / "different-source"
    different.mkdir()
    scores = pd.read_parquet(inputs[7])
    scores.loc[:, ["p_positive", "p_negative"]] = scores[["p_negative", "p_positive"]].to_numpy()
    scores["sentiment_score"] *= -1
    _, exported, _ = _export(different, scores, inputs[0])
    ablation = replace(inputs[5], scored_articles_path=str(different / "features-scored.parquet"))
    changed = _plan(inputs, cache=cache, sentiment_export=exported, ablation=ablation)
    assert changed["metadata"]["news_export_sha256"] != _plan(inputs, cache=cache)["metadata"]["news_export_sha256"]
    assert counts == dict(prepare=2, articles=2, article_hash=3, prefix=12, scaler=12)


def test_corrupt_scored_source_cannot_bypass_checksum_using_cached_articles(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    counts = _spies(monkeypatch)
    cache = {}
    _plan(inputs, cache=cache)
    source = inputs[7]
    source.write_bytes(source.read_bytes() + b"corruption after caching")
    target = tmp_path / "refused"
    with pytest.raises(ValueError, match="checksum"):
        news.run_news_sentiment_ablation(
            inputs[0], inputs[2], inputs[3], inputs[4], [1], target,
            sentiment_export=inputs[1], ablation=inputs[5], n_splits=2, gap_bars=2,
            dry_run=True, _study_cache=cache,
        )
    assert counts["articles"] == 1 and counts["article_hash"] == 2
    assert counts["prepare"] == 2
    assert not target.exists()


def test_checkpoint_change_cannot_reuse_article_validation(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    counts = _spies(monkeypatch)
    cache = {}
    _plan(inputs, cache=cache)
    exported = replace(inputs[1], manifest={**inputs[1].manifest, "checkpoint": "different-checkpoint"})
    with pytest.raises(ValueError, match="checkpoint"):
        _plan(inputs, cache=cache, sentiment_export=exported)
    assert counts["articles"] == 2 and counts["article_hash"] == 2


def test_news_manifest_change_invalidates_exports_and_scalers_but_not_prices(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    counts = _spies(monkeypatch)
    cache = {}
    original = _plan(inputs, cache=cache)
    exported = replace(inputs[1], manifest={**inputs[1].manifest, "extra_audit": "new metadata"})
    changed = _plan(inputs, cache=cache, sentiment_export=exported)
    assert counts == dict(prepare=2, articles=1, article_hash=2, prefix=12, scaler=12)
    assert original["task_specs"][0]["signature"] != changed["task_specs"][0]["signature"]


def test_required_scaler_policy_is_part_of_cache_identity(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    counts = _spies(monkeypatch)
    cache = {}
    _plan(inputs, cache=cache, ablation=replace(inputs[5], candidates=("gru",)))
    assert counts["scaler"] == 2
    _plan(inputs, cache=cache, ablation=replace(inputs[5], candidates=("gru", "gru_activity")))
    assert counts["prepare"] == 2
    assert counts["scaler"] == 4
    assert not cache["prefix_exports"], "A full uncontrolled export must not enter the TRAIN/INNER cache."


def test_study_cache_never_opens_outer_before_checkpoint_and_results_match(tmp_path, monkeypatch):
    import torch

    inputs = _inputs(tmp_path)
    market, exported, config, loss, parameters, ablation, _, _ = inputs
    ablation = replace(ablation, candidates=("gru_features_shuffled",))
    options = dict(sentiment_export=exported, ablation=ablation, n_splits=2, gap_bars=2)
    cache = {}
    cached_target, uncached_target = tmp_path / "cached", tmp_path / "uncached"
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    current_target = cached_target
    original_control = fnspid_polarity.prepare_fold_news_control
    original_splits = news._prepare_splits
    checkpoints = []
    original_save = news.atomic_torch_save

    def save(path, value):
        result = original_save(path, value)
        checkpoints.append(Path(path))
        return result

    def control(exported, scored, fold, **kwargs):
        if kwargs["include_outer"]:
            assert any(path.parent == current_target and path.name.startswith(f"fold-{fold['fold']}-")
                       for path in checkpoints), "An OUTER news control requires a frozen checkpoint."
        return original_control(exported, scored, fold, **kwargs)

    def splits(*args, **kwargs):
        if kwargs.get("include_test"):
            assert checkpoints and checkpoints[-1].parent == current_target
            assert checkpoints[-1].exists(), "OUTER price preparation requires a saved checkpoint."
        return original_splits(*args, **kwargs)

    monkeypatch.setattr(news, "atomic_torch_save", save)
    monkeypatch.setattr(news, "_prepare_splits", splits)
    monkeypatch.setattr(fnspid_polarity, "prepare_fold_news_control", control)
    try:
        news.run_news_sentiment_ablation(market, config, loss, parameters, [1], cached_target,
                                        **options, _study_cache=cache, dry_run=True)
        assert not checkpoints and not cached_target.exists()
        cached = news.run_news_sentiment_ablation(market, config, loss, parameters, [1], cached_target,
                                                  **options, _study_cache=cache)
        current_target = uncached_target
        uncached = news.run_news_sentiment_ablation(market, config, loss, parameters, [1], uncached_target,
                                                    **options)
        for left, right in zip(cached["folds"], uncached["folds"], strict=True):
            assert left["task_signature"] == right["task_signature"]
            assert left["outer_metrics"] == right["outer_metrics"]
            assert left["inner_metrics"] == right["inner_metrics"]
            assert left["sentiment_scaler"] == right["sentiment_scaler"]
            for partition in ("inner", "outer"):
                pd.testing.assert_frame_equal(
                    pd.read_parquet(cached_target / left["prediction_artifacts"][partition]),
                    pd.read_parquet(uncached_target / right["prediction_artifacts"][partition]),
                    check_exact=True,
                )
        assert len(checkpoints) == 4
        for prefix in cache["prefix_exports"].values():
            control = prefix.manifest["polarity_control"]
            assert control["include_outer"] is False
            assert "outer" not in control["decision_intervals"]
            inner_end = pd.Timestamp(control["decision_intervals"]["inner"]["end"])
            assert prefix.frame.date.lt(inner_end).all()
        assert set(cache) == {"control_articles", "price_prepared", "prefix_exports", "prefix_scalers"}
    finally:
        torch.set_num_threads(old_threads)
