"""Offline, matched news controls with a sealed final calendar holdout.

This runner consumes verified daily exports, never collects news or scores text.
Financial objectives use the complete price calendar; an unavailable standalone
sentiment signal explicitly takes a flat position, not a fabricated prediction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import _nullable_metadata, hash_dataframe
from trading_system.artifacts.multimodal_study import (
    atomic_torch_save, atomic_write_json, completion_manifest, file_record,
    preprocessing_state, stable_digest, training_signature,
)
from trading_system.data.multimodal import build_multimodal_dataset
from trading_system.data.news_sentiment import (
    SENTIMENT_COLUMNS, SENTIMENT_STATISTICS, SentimentExport,
)
from trading_system.data.fnspid_export import (
    AVAILABILITY_ASSUMPTION, EXPLORATORY_LIMITATIONS, FNSPID_PROTOCOL, FNSPID_SCHEMA,
    _scored_articles, _sha256,
)
from trading_system.models.multimodal_branches import GRUBranch, SentimentBranch
from trading_system.models.multimodal_system import MaskedLogitFusion
from trading_system.models.neural.config import GRUConfig
from trading_system.models.neural.trainer import resolve_device, seed_torch_run
from trading_system.training.financial_loss import ReturnPanel, position_coefficients
from trading_system.training.learning_trace import LearningTrace, gradient_norm_l2
from .graph_ablation import (
    GraphAblationConfig, _atomic_parquet, _classification, _complete, _daily_paths,
    _dates, _fit, _prepare, _resume_metadata, _run_context as _graph_context,
    _scale, _validated_row,
)
from .runner import _prepare_splits


CANDIDATES = ("gru", "gru_activity", "gru_features", "sentiment", "gru_sentiment_mean")
POLARITY_CANDIDATES = ("gru", "gru_activity", "gru_features", "gru_features_shuffled", "gru_features_neutralized")
ALL_CANDIDATES = tuple(dict.fromkeys((*CANDIDATES, *POLARITY_CANDIDATES)))
POLARITY_STATISTICS = tuple(name for name in SENTIMENT_STATISTICS
                            if name not in ("confidence_mean", "hours_since_last_news"))
_FEATURE_CANDIDATES = ("gru_features", "gru_features_shuffled", "gru_features_neutralized")
_PREFIX = "news_sentiment__"
_COVERED_COLUMN = _PREFIX + "covered"
_OBSERVED_COLUMN = _PREFIX + "observed"
_DATASET_SENTIMENT_COLUMNS = ("_scaled_news_count", *SENTIMENT_COLUMNS[1:])


@dataclass(frozen=True)
class NewsSentimentAblationConfig:
    candidates: tuple[str, ...] = CANDIDATES
    sentiment_hidden_size: int = 16
    date_batch_size: int = 32
    news_protocol: str = "pit"
    shuffle_seed: int = 314159
    scored_articles_path: str | None = None
    learning_diagnostics: bool = False

    def __post_init__(self):
        if not isinstance(self.learning_diagnostics, bool):
            raise ValueError("learning_diagnostics must be a bool.")
        if self.news_protocol not in ("pit", FNSPID_PROTOCOL):
            raise ValueError("news_protocol must be pit or fnspid-exploratory.")
        if (not self.candidates or len(set(self.candidates)) != len(self.candidates)
                or any(item not in ALL_CANDIDATES for item in self.candidates)):
            raise ValueError(f"Choose unique news candidates from {ALL_CANDIDATES}.")
        if isinstance(self.shuffle_seed, bool) or not isinstance(self.shuffle_seed, int) or not 0 <= self.shuffle_seed < 2**32:
            raise ValueError("shuffle_seed must be an integer in [0, 2**32).")
        if any(item in self.candidates for item in POLARITY_CANDIDATES[3:]) and self.news_protocol != FNSPID_PROTOCOL:
            raise ValueError("Article polarity controls require the explicit fnspid-exploratory protocol.")
        for name in ("sentiment_hidden_size", "date_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

    @property
    def graph_lookback(self):
        # _prepare shares a price-window warmup with the graph runner. There is
        # no graph here; its valid minimal lookback avoids a 252-bar news tax.
        return 3


def _align_news(frame, exported, config):
    """Left-align exact row keys, including missing export decisions."""
    keys = pd.MultiIndex.from_arrays((_dates(frame, config.date_col), frame[config.group_col]),
                                     names=("date", "ticker"))
    source = exported.frame.copy()
    source["date"] = _dates(source, "date")
    source["export_row_present"] = True
    source = source.set_index(["date", "ticker"])
    aligned = source.reindex(keys).reset_index()
    aligned["export_row_present"] = aligned["export_row_present"].eq(True).fillna(False).astype(bool)
    aligned["source_available"] = aligned["source_available"].eq(True).fillna(False).astype(bool)
    aligned["coverage_status"] = aligned["coverage_status"].fillna("unknown")
    if exported.protocol == FNSPID_PROTOCOL:
        aligned["observation_status"] = aligned["observation_status"].fillna("unobserved")
        aligned["availability_kind"] = aligned["availability_kind"].fillna(AVAILABILITY_ASSUMPTION)
    aligned["available_at"] = pd.to_datetime(aligned["available_at"], utc=True)
    return aligned


@dataclass(frozen=True)
class _SentimentScaler:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    covered_fit_rows: int
    feature_fit_rows: tuple[int, ...]
    news_protocol: str = "pit"

    @classmethod
    def fit(cls, prepared, exported, config, *, required):
        aligned = _align_news(prepared.train, exported, config)
        eligible = prepared.train["_fit_eligible"].to_numpy(dtype=bool)
        covered = aligned.source_available.to_numpy(dtype=bool) & eligible
        if required and not covered.any():
            if exported.protocol == FNSPID_PROTOCOL:
                raise ValueError("No observed eligible TRAIN news under the FNSPID publication-delay assumption.")
            raise ValueError("No covered eligible TRAIN news observations; historical backfills collected now "
                             "must remain unavailable. Supply point-in-time coverage, not dummy zeros.")
        means, scales, counts = [], [], []
        for column in ("news_count", *SENTIMENT_STATISTICS):
            valid = covered.copy()
            if column != "news_count":
                valid &= aligned[f"{column}_present"].fillna(0).to_numpy(dtype=bool)
            values = aligned.loc[valid, column].to_numpy(dtype=np.float64)
            if len(values) and not np.isfinite(values).all():
                raise ValueError(f"Non-finite covered TRAIN {column}.")
            mean, scale = (float(values.mean()), float(values.std())) if len(values) else (0., 1.)
            means.append(mean)
            scales.append(scale if scale > 0 else 1.)
            counts.append(int(len(values)))
        return cls(tuple(means), tuple(scales), int(covered.sum()), tuple(counts), exported.protocol)

    def transform(self, aligned):
        result = aligned.copy()
        covered = result.source_available.to_numpy(dtype=bool)
        for index, column in enumerate(("news_count", *SENTIMENT_STATISTICS)):
            valid = covered.copy()
            if column != "news_count":
                presence = result[f"{column}_present"].fillna(0).to_numpy(dtype=np.float32)
                valid &= presence.astype(bool)
                result[f"{column}_present"] = np.where(covered, presence, 0).astype(np.float32)
            values = result[column].fillna(0).to_numpy(dtype=np.float64)
            result[column] = np.where(valid, (values - self.means[index]) / self.scales[index], 0).astype(np.float32)
        return result

    def state_dict(self):
        exploratory = self.news_protocol == FNSPID_PROTOCOL
        return {"columns": ["news_count", *SENTIMENT_STATISTICS], "mean": self.means,
                "scale": self.scales, "covered_fit_rows": 0 if exploratory else self.covered_fit_rows,
                "available_fit_rows": self.covered_fit_rows,
                "observed_fit_rows": self.covered_fit_rows if exploratory else None,
                "news_protocol": self.news_protocol,
                "feature_fit_rows": self.feature_fit_rows,
                "fit_policy": ("observed eligible TRAIN rows under the publication-delay assumption; statistics require presence"
                               if exploratory else "covered eligible TRAIN rows only; statistics require presence"),
                "presence_policy": "binary channels unchanged on available rows; zero when unavailable"}


def _temporal_columns(prepared, candidate, news_protocol="pit"):
    indicator = _OBSERVED_COLUMN if news_protocol == FNSPID_PROTOCOL else _COVERED_COLUMN
    if candidate == "gru_activity":
        return (*prepared.columns, _PREFIX + "news_count", indicator)
    if candidate in _FEATURE_CANDIDATES:
        return (*prepared.columns, *(_PREFIX + name for name in SENTIMENT_COLUMNS), indicator)
    return prepared.columns


def _dataset(target, history, prepared, exported, scaler, config, candidate):
    columns = _temporal_columns(prepared, candidate, exported.protocol)
    raw_news = _align_news(target, exported, config)
    news = scaler.transform(raw_news)

    def model_features(aligned):
        # Disable value channels in standardized space, not by claiming that
        # real articles were neutral. Keep dimensions and all presence masks.
        if candidate == "gru_features_neutralized":
            aligned = aligned.copy()
            aligned.loc[:, list(POLARITY_STATISTICS)] = np.float32(0.)
        return aligned

    news = model_features(news)

    def append(frame, aligned):
        result = frame.copy()
        for name in SENTIMENT_COLUMNS:
            result[_PREFIX + name] = aligned[name].to_numpy(dtype=np.float32)
        indicator = _OBSERVED_COLUMN if exported.protocol == FNSPID_PROTOCOL else _COVERED_COLUMN
        result[indicator] = aligned.source_available.to_numpy(dtype=np.float32)
        return result

    if candidate in ("gru_activity", *_FEATURE_CANDIDATES):
        target = append(target, news)
        history = append(history, model_features(scaler.transform(_align_news(history, exported, config))))
    source = news.drop(columns="export_row_present").rename(
        columns={"date": config.date_col, "ticker": config.group_col})
    # The multimodal contract verifies covered empty windows using RAW count.
    # The neural channel may be standardized, so keep these roles separate.
    source["_scaled_news_count"] = source["news_count"]
    source["news_count"] = raw_news["news_count"].fillna(0).to_numpy(dtype=np.float32)
    dataset = build_multimodal_dataset(
        target, tickers=prepared.tickers, context_len=config.context_len,
        temporal_columns=columns, node_columns=prepared.columns,
        history_frame=history, sentiment_frame=source, sentiment_columns=_DATASET_SENTIMENT_COLUMNS,
        sentiment_protocol=exported.protocol,
        date_col=config.date_col, ticker_col=config.group_col,
    )
    for batch in dataset.iter_batches(128):
        if not batch.asset_mask.all() or not batch.temporal_mask.all():
            raise ValueError("News ablation requires the complete matched price GRU calendar.")
    return dataset


def _make_model(sample, candidate, config, parameters, ablation, seed, torch):
    training = GRUConfig(**{**parameters, "seed": seed, "device": config.device})
    if candidate == "sentiment":
        model = SentimentBranch(len(SENTIMENT_COLUMNS), hidden_size=ablation.sentiment_hidden_size)
    elif candidate == "gru_sentiment_mean":
        class MeanNews(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = GRUBranch(sample.temporal.shape[-1], config.context_len, training)
                self.sentiment = SentimentBranch(len(SENTIMENT_COLUMNS), hidden_size=ablation.sentiment_hidden_size)
                self.fusion = MaskedLogitFusion(("gru", "sentiment"), mode="mean")

            def forward(self, batch):
                output, _ = self.fusion({"gru": self.gru(batch), "sentiment": self.sentiment(batch)})
                return output
        model = MeanNews()
    else:
        model = GRUBranch(sample.temporal.shape[-1], config.context_len, training)
    return model.to(resolve_device(config.device, torch))


def _batch_predictions(model, batch, config, torch):
    output = model(batch)
    classes = torch.softmax(output.logits, dim=-1)
    flat = torch.zeros_like(classes)
    flat[..., 1] = 1.
    # Availability is never changed. Only the executable policy has a defined
    # fallback; logits from missing standalone rows are not scored or exported.
    classes = torch.where(output.availability[..., None], classes, flat)
    coefficients = torch.as_tensor(position_coefficients(config.resolved_backtest_position_mode()),
                                   dtype=classes.dtype, device=classes.device)
    return output, classes, classes @ coefficients


def _masked_positions(model, dataset, config, ablation, torch, backward=None):
    values = np.zeros(len(dataset) * len(dataset.tickers), dtype=np.float64)
    for batch in dataset.iter_batches(ablation.date_batch_size):
        _, _, positions = _batch_predictions(model, batch, config, torch)
        rows = batch.row_positions.reshape(-1)
        if backward is None:
            values[rows] = positions.reshape(-1).detach().cpu().numpy()
        else:
            gradient = torch.as_tensor(backward[rows].reshape(positions.shape),
                                       dtype=positions.dtype, device=positions.device)
            positions.backward(gradient)
    return values


def _fit_sentiment(model, train_ds, val_ds, train_panel, val_panel, loss, config, training, ablation, torch):
    """Same two-pass ReturnPanel gradient and checkpoint rule, allowing FLAT."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate,
                                  weight_decay=training.weight_decay)
    best, state, best_epoch, stale = np.inf, None, 0, 0
    started = perf_counter()
    trace = LearningTrace() if ablation.learning_diagnostics else None
    stop_reason = "max_epochs"
    for epoch in range(training.epochs):
        epoch_seed = (training.seed + epoch + 1) % (2**32)
        seed_torch_run(epoch_seed, training.deterministic, torch)
        model.train()
        with torch.no_grad():
            positions = _masked_positions(model, train_ds, config, ablation, torch)
        train_loss, gradient = train_panel.loss_and_gradient(positions, loss)
        train_phase = LearningTrace.phase(train_loss, positions) if trace is not None else None
        optimizer.zero_grad(set_to_none=True)
        seed_torch_run(epoch_seed, training.deterministic, torch)
        _masked_positions(model, train_ds, config, ablation, torch, backward=gradient)
        gradient_norm = gradient_norm_l2(model.parameters()) if trace is not None else None
        if training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm)
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError("Non-finite sentiment financial gradient.")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            positions = _masked_positions(model, val_ds, config, ablation, torch)
        score, _ = val_panel.loss_and_gradient(positions, loss)
        validation_phase = LearningTrace.phase(score, positions) if trace is not None else None
        improved = score < best - training.early_stopping_min_delta
        if improved:
            best, state, best_epoch, stale = score, deepcopy(model.state_dict()), epoch + 1, 0
        else:
            stale += 1
        if trace is not None:
            trace.record(epoch + 1, train=train_phase, validation=validation_phase,
                         gradient_norm_pre_clip=gradient_norm, improved=improved,
                         stale=stale, best_epoch=best_epoch)
        if stale >= training.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    if state is None:
        raise RuntimeError("No finite sentiment checkpoint.")
    model.load_state_dict(state)
    fitted = {"best_epoch": best_epoch, "epochs_run": epoch + 1, "seconds": perf_counter() - started,
              "parameter_count": sum(p.numel() for p in model.parameters())}
    if trace is not None:
        fitted["learning_trace"] = trace.finish(stop_reason=stop_reason, best_epoch=best_epoch)
    return fitted


def _coverage(frame, exported, config):
    aligned = _align_news(frame, exported, config)
    covered = aligned.coverage_status.eq("covered").to_numpy(dtype=bool)
    observed = aligned.source_available.to_numpy(dtype=bool)
    count = aligned.news_count.to_numpy(dtype=float)
    return {"rows": len(aligned), "covered_rows": int(covered.sum()),
            "covered_fraction": float(covered.mean()),
            "covered_zero_news_rows": int((covered & (count == 0)).sum()),
            "covered_article_rows": int((covered & (count > 0)).sum()),
            "incomplete_rows": int(aligned.coverage_status.eq("incomplete").sum()),
            "unknown_rows": int(aligned.coverage_status.eq("unknown").sum()),
            "missing_export_rows": int((~aligned.export_row_present).sum()),
            "signal_available_rows": int(observed.sum()),
            "observed_rows": int((observed & (count > 0)).sum()),
            "unobserved_rows": int((~observed).sum()),
            "news_protocol": exported.protocol,
            "by_ticker": [{"ticker": name, "rows": int(len(part)),
                           "covered_rows": int(part.coverage_status.eq("covered").sum()),
                           "signal_available_rows": int(part.source_available.sum())}
                          for name, part in aligned.groupby("ticker", sort=True)]}


def _export_predictions(model, dataset, frame, path, *, candidate, fold, seed, partition,
                        exported, config, ablation, torch):
    import os
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    news = _align_news(frame, exported, config)
    positions = np.zeros(len(frame), dtype=np.float64)
    probabilities = np.zeros((len(frame), 3), dtype=np.float64)
    available = np.zeros(len(frame), dtype=bool)
    seen = np.zeros(len(frame), dtype=bool)
    handle, temporary = tempfile.mkstemp(prefix=f".{Path(path).name}.", dir=Path(path).parent)
    os.close(handle)
    writer = None
    try:
        for batch in dataset.iter_batches(ablation.date_batch_size):
            output, classes, value = _batch_predictions(model, batch, config, torch)
            rows = batch.row_positions.reshape(-1)
            if (rows < 0).any() or (rows >= len(frame)).any() or seen[rows].any():
                raise ValueError("Duplicate or invalid news prediction/backtest keys.")
            logits = output.logits.reshape(-1, 3).detach().cpu().numpy()
            probs = classes.reshape(-1, 3).detach().cpu().numpy()
            usable = output.availability.reshape(-1).detach().cpu().numpy()
            values = value.reshape(-1).detach().cpu().numpy()
            if not np.isfinite(logits).all() or not np.isfinite(probs).all():
                raise FloatingPointError("Non-finite news predictions.")
            logits = np.where(usable[:, None], logits, np.nan)
            positions[rows], probabilities[rows], available[rows], seen[rows] = values, probs, usable, True
            part, source = frame.iloc[rows], news.iloc[rows]
            days = pd.to_datetime(part[config.date_col], utc=True).reset_index(drop=True)
            tickers = part[config.group_col].astype(str).reset_index(drop=True)
            known = part["_label_known"].to_numpy(dtype=bool)
            result = pd.DataFrame({
                "variant_id": candidate, "candidate": candidate, "fold": fold, "seed": seed,
                "partition": partition, "date": days, "ticker": tickers,
                "backtest_key": days.map(lambda day: day.isoformat()) + "|" + tickers,
                "row_position": rows, "available": usable, "news_available": source.source_available.to_numpy(),
                "coverage_status": source.coverage_status.to_numpy(),
                "news_protocol": exported.protocol,
                "observation_status": source.get("observation_status", pd.Series(
                    np.where(source.news_count.gt(0), "observed", "unobserved"), index=source.index
                )).to_numpy(),
                "availability_kind": source.get("availability_kind", pd.Series("audited_point_in_time", index=source.index)).to_numpy(),
                "export_row_present": source.export_row_present.to_numpy(),
                "news_count": source.news_count.to_numpy(dtype=float),
                "available_at": pd.to_datetime(source.available_at, utc=True).reset_index(drop=True),
                "label_known": known, "label": np.where(known, part.Label_id.to_numpy(dtype=np.int64), -1),
                "adj_close": part[config.price_col].to_numpy(dtype=float),
                "logit_sell": logits[:, 0], "logit_hold": logits[:, 1], "logit_buy": logits[:, 2],
                "p_sell": probs[:, 0], "p_hold": probs[:, 1], "p_buy": probs[:, 2],
                "prediction": probs.argmax(axis=1), "position": values,
                "probability_kind": np.where(usable, "softmax", "flat_fallback"),
            })
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
        if not seen.all():
            raise ValueError("Predictions lost matched market rows.")
        writer.close()
        writer = None
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    return positions, probabilities, available


def _context(frame, config, loss, parameters, seeds, exported, ablation, **options):
    if not isinstance(exported, SentimentExport) or tuple(exported.columns) != SENTIMENT_COLUMNS:
        raise TypeError("sentiment_export must be a verified SentimentExport with the 23 standard features.")
    exploratory = ablation.news_protocol == FNSPID_PROTOCOL
    if exported.protocol != ablation.news_protocol:
        raise ValueError("News ablation protocol and verified export disagree; choose the explicit matching protocol.")
    if exploratory:
        if (exported.manifest.get("schema_version") != FNSPID_SCHEMA
                or exported.manifest.get("protocol") != FNSPID_PROTOCOL
                or exported.manifest.get("point_in_time") is not False
                or exported.manifest.get("historical_coverage_claim") is not False
                or exported.manifest.get("availability_kind") != AVAILABILITY_ASSUMPTION):
            raise ValueError("FNSPID exploratory manifest must declare its assumptions and absent historical coverage.")
    elif exported.manifest.get("schema_version") != "1.0" or exported.manifest.get("point_in_time") is False:
        raise ValueError("Strict PIT ablation requires an audited point-in-time export.")
    if loss.objective not in ("sharpe", "combined"):
        raise ValueError("News comparison supports exactly Sharpe or combined financial loss.")
    if config.feature_set == "expanded" and "sentiment" in config.expanded_feature_groups:
        raise ValueError("Exclude the legacy sentiment group from the frozen price-feature pool.")
    if any(name.startswith(_PREFIX) for name in frame.columns):
        raise ValueError("Price data already contains reserved news comparison feature columns.")
    identifiers = exported.manifest.get("input_identifiers", {})
    if not isinstance(identifiers, dict):
        raise ValueError("News export input_identifiers must be an object.")
    domain = identifiers.get(
        "domain", exported.manifest.get("domain", exported.manifest.get("target_domain", "ticker")))
    if domain not in ("ticker", "company", "equity"):
        raise ValueError("Macro news must be routed to a separate market-domain model, not ticker sentiment.")
    work, config, folds, metadata = _graph_context(
        frame, config, loss, parameters, seeds,
        GraphAblationConfig(graph_lookback=3, candidates=("gru",)), **options,
    )
    extras = set(exported.frame.ticker) - set(work[config.group_col])
    if extras:
        raise ValueError(f"News export has non-selected or macro tickers: {sorted(extras)}. Export ticker-domain decisions only.")
    source = exported.frame
    if source.duplicated(["date", "ticker"]).any():
        raise ValueError("News export has duplicate decision keys.")
    if not exploratory and not np.array_equal(source.source_available.to_numpy(dtype=bool), source.coverage_status.eq("covered").to_numpy()):
        raise ValueError("News export source availability disagrees with coverage.")
    dates = pd.to_datetime(source.date, utc=True, errors="raise")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("News decision points must be midnight UTC.")
    # The path-based CLI has already verified Parquet and its manifest. Also
    # enforce the causal batch contract on programmatic inputs, during dry-run
    # and before creating any run directory or training a model.
    build_multimodal_dataset(source[["date", "ticker"]], tickers=tuple(sorted(set(source.ticker))),
        context_len=1, sentiment_frame=source, sentiment_columns=SENTIMENT_COLUMNS,
        sentiment_protocol=ablation.news_protocol)
    for name in ("graph_context_path", "graph_context_sha256", "sector_context_columns",
                 "market_context_path", "market_context_sha256", "market_columns", "market_audit"):
        metadata.pop(name, None)
    ablation_metadata = asdict(ablation)
    if not ablation.learning_diagnostics:
        # Keep the legacy opt-out metadata/spec shape for existing runs.
        ablation_metadata.pop("learning_diagnostics")
    metadata.update(
        protocol="matched_purged_news_sentiment_ablation", ablation=ablation_metadata,
        news_protocol=ablation.news_protocol, point_in_time=not exploratory,
        historical_availability_verified=not exploratory,
        availability_kind=AVAILABILITY_ASSUMPTION if exploratory else "audited_point_in_time",
        warnings=EXPLORATORY_LIMITATIONS if exploratory else [],
        news_export_sha256=hash_dataframe(source), news_manifest=exported.manifest,
        news_manifest_sha256=stable_digest(exported.manifest), news_columns=list(SENTIMENT_COLUMNS),
        shared_price_warmup=3,
        export_contract={**metadata["export_contract"],
            "missing_sentiment": "standalone FLAT, hold fallback probabilities, null logits, available=false; GRU never masked",
            "mean_fusion": "joint financial training; available-branch masked mean logits; not calibrated confidence fusion"},
        limitations=["No news collection, article scoring, download or historical availability reconstruction.",
                     "Covered zero-news is available evidence; unknown/incomplete/missing coverage is unavailable.",
                     "No macro news forced into company/ticker inputs.",
                     "Mean-logit fusion is not a calibration or confidence claim.",
                     "Current-universe survivorship and upstream coverage biases remain.",
                     "Fold/seed averages are not independent market paths or a concatenated backtest."],
    )
    if exploratory:
        metadata["limitations"] = [*EXPLORATORY_LIMITATIONS,
            "No macro news forced into company/ticker inputs.",
            "Mean-logit fusion is not a calibration or confidence claim.",
            "Fold/seed averages are not independent market paths or a concatenated backtest."]
    return work, config, folds, _nullable_metadata(metadata)


def _control_mode(candidate):
    return {"gru_features_shuffled": "shuffled", "gru_features_neutralized": "neutralized"}.get(candidate, "original")


def _load_control_articles(exported, ablation, *, _study_cache=None):
    if "gru_features_shuffled" not in ablation.candidates:
        return None, None
    if not ablation.scored_articles_path:
        raise ValueError("Article shuffling requires --news-scored-articles from the verified preparation.")
    path = Path(ablation.scored_articles_path).expanduser().resolve(strict=True)
    record = exported.manifest.get("inputs", {}).get("scored", {})
    digest = _sha256(path)
    if digest != record.get("sha256"):
        raise ValueError("Scored article checksum differs from the frozen daily news export.")
    key = stable_digest({"sha256": digest, "checkpoint": exported.manifest["checkpoint"],
                         "protocol": exported.protocol})
    cached = _study_cache.setdefault("control_articles", {}) if _study_cache is not None else None
    if cached is not None and key in cached:
        scored = cached[key]
    else:
        scored = _scored_articles(path, exported.manifest["checkpoint"])
        if cached is not None:
            cached[key] = scored
    if len(scored) != record.get("rows"):
        raise ValueError("Scored article count differs from the frozen daily news export.")
    return scored, {"path": str(path), "sha256": digest, "rows": len(scored)}


def _study_prepared_key(metadata, fold, ablation):
    # Candidate lists and corpus seeds cannot affect price preprocessing. Every
    # effective data/config/calendar/CV/runtime dependency still participates.
    # training_signature excludes observed_torch_state, which training changes
    # itself, while retaining source bytes and numerical runtime settings.
    return training_signature({
        "cache_schema_version": 1, "dataset_sha256": metadata["dataset_sha256"],
        "config": metadata["config"], "loss_config": metadata["loss_config"],
        "gru_parameters": metadata["gru_parameters"], "calendar": metadata["calendar"],
        "cv_spec": metadata["cv_spec"], "provenance": metadata["provenance"],
        "fold": {"fold": fold["fold"], "split": asdict(fold["split"]), "end": fold["end"]},
        "shared_price_warmup": ablation.graph_lookback,
    })


def _study_news_key(metadata, prepared_key, mode, ablation, controlled):
    return stable_digest({
        "cache_schema_version": 1, "prepared_key": prepared_key, "mode": mode,
        "controlled": controlled, "shuffle_seed": ablation.shuffle_seed if mode == "shuffled" else None,
        "news_protocol": metadata["news_protocol"],
        "news_export_sha256": metadata["news_export_sha256"],
        "news_manifest_sha256": metadata["news_manifest_sha256"],
        "scored_sha256": metadata["news_manifest"].get("inputs", {}).get("scored", {}).get("sha256"),
        "checkpoint": metadata["news_manifest"].get("checkpoint"),
        "include_outer": False,
    })


def _control_info(exported, candidate):
    return {**exported.manifest.get("polarity_control", {}), "mode": _control_mode(candidate),
            "export_sha256": hash_dataframe(exported.frame),
            "neutralized_statistics": list(POLARITY_STATISTICS) if candidate == "gru_features_neutralized" else [],
            "neutralization_applied": candidate == "gru_features_neutralized",
            "neutralization_policy": ("zero standardized numeric channels; preserve dimensions, presence, activity, recency and confidence"
                                      if candidate == "gru_features_neutralized" else None)}


def _task_spec(metadata, prepared, scaler, fold, candidate, seed, control=None):
    snapshot = preprocessing_state(prepared)
    snapshot["sentiment_scaler"] = scaler.state_dict()
    snapshot["effective_temporal_columns"] = list(_temporal_columns(prepared, candidate, metadata["news_protocol"]))
    if control is not None:
        snapshot["news_control"] = control
    return _nullable_metadata({
        "schema_version": 1, "candidate": candidate, "fold": fold["fold"], "seed": seed,
        "config": metadata["config"], "loss_config": metadata["loss_config"],
        "gru_parameters": metadata["gru_parameters"], "ablation": metadata["ablation"],
        "dataset_sha256": metadata["dataset_sha256"], "calendar": metadata["calendar"],
        "news_export_sha256": metadata["news_export_sha256"],
        "news_manifest_sha256": metadata["news_manifest_sha256"], "news_columns": metadata["news_columns"],
        "fold_boundaries": {"split": asdict(fold["split"]), "end": fold["end"]},
        "cv_spec": metadata["cv_spec"], "preprocessing": snapshot,
        "model": {"candidate": candidate, "temporal_width": len(_temporal_columns(prepared, candidate, metadata["news_protocol"])),
                  "sentiment_width": len(SENTIMENT_COLUMNS),
                  "sentiment_hidden_size": metadata["ablation"]["sentiment_hidden_size"],
                  "fusion": "masked_mean_logits" if candidate == "gru_sentiment_mean" else None},
        "eligible_sessions": {name: [day.isoformat() for day in _dates(value, metadata["config"]["date_col"]).unique().sort_values()]
                              for name, value in (("train", prepared.train), ("inner", prepared.validation))},
        "resolved_device": str(resolve_device(metadata["config"]["device"], __import__("torch"))),
        "precision": "float32", "provenance": metadata["provenance"],
    })


def plan_run_news_sentiment_ablation(frame, config, loss, gru_parameters, seeds, *, sentiment_export,
                                     ablation=NewsSentimentAblationConfig(), n_splits=3,
                                     initial_train_fraction=.5, inner_val_fraction=.2,
                                     gap_bars=5, embargo_bars=0, dataset_path=None, _runtime_cache=None,
                                     _study_cache=None):
    """Read-only train preprocessing and exact task signatures; no models fit.

    A private study cache may share verified article reads and TRAIN/INNER price
    preparations, exports and scalers between matched comparison calls. Context,
    hashes and signatures are revalidated each call; OUTER is never cached here.
    """
    if _study_cache is not None and not isinstance(_study_cache, dict):
        raise TypeError("_study_cache must be a local dictionary or None.")
    work, config, folds, metadata = _context(
        frame, config, loss, gru_parameters, seeds, sentiment_export, ablation,
        n_splits=n_splits, initial_train_fraction=initial_train_fraction,
        inner_val_fraction=inner_val_fraction, gap_bars=gap_bars,
        embargo_bars=embargo_bars, dataset_path=dataset_path,
    )
    controlled = any(candidate in POLARITY_CANDIDATES[3:] for candidate in ablation.candidates)
    scored, article_record = _load_control_articles(sentiment_export, ablation, _study_cache=_study_cache)
    if controlled:
        metadata["polarity_controls"] = {
            "schema_version": 1, "shuffle_seed": ablation.shuffle_seed,
            "scored_articles": article_record,
            "shuffle": "Joint article probability/score/confidence permutation within ticker and calendar partition, before daily aggregation.",
            "neutralized_statistics": list(POLARITY_STATISTICS),
            "neutralization": "Zero standardized numeric channels, not fake neutral articles; identical model width and presence masks.",
            "preserved": ["calendar", "tickers", "prices", "labels", "news_count", "recency", "availability", "presence_masks"],
            "limitations": ["Permutation can use a later score within the same partition: a retrospective null control, not a deployable signal.",
                            "Shuffling also perturbs confidence alignment; neutralization retains confidence information.",
                            "One frozen permutation seed across model seeds; these are not independent corpus permutations."],
        }
    tasks = []
    prepared_folds = {}
    price_cache = _study_cache.setdefault("price_prepared", {}) if _study_cache is not None else None
    export_cache = _study_cache.setdefault("prefix_exports", {}) if _study_cache is not None else None
    scaler_cache = _study_cache.setdefault("prefix_scalers", {}) if _study_cache is not None else None
    required = any(candidate != "gru" for candidate in ablation.candidates)
    for fold in folds:
        prepared_key = _study_prepared_key(metadata, fold, ablation) if price_cache is not None else None
        if price_cache is not None and prepared_key in price_cache:
            prepared = price_cache[prepared_key]
        else:
            fold_frame = work.loc[_dates(work, config.date_col) <= pd.Timestamp(fold["end"])].copy()
            prepared = _prepare(fold_frame, replace(config, purged_split=fold["split"]), ablation)
            if price_cache is not None:
                price_cache[prepared_key] = prepared
        modes = dict.fromkeys(_control_mode(candidate) for candidate in ablation.candidates) if controlled else {"original": None}
        exports, scalers = {}, {}
        for mode in modes:
            news_key = _study_news_key(metadata, prepared_key, mode, ablation, controlled) if export_cache is not None else None
            if controlled and export_cache is not None and news_key in export_cache:
                exported = export_cache[news_key]
            elif controlled:
                from trading_system.data.fnspid_polarity import prepare_fold_news_control
                exported = prepare_fold_news_control(sentiment_export, scored, fold, mode=mode,
                            shuffle_seed=ablation.shuffle_seed, include_outer=False)
                if export_cache is not None:
                    export_cache[news_key] = exported
            else:
                exported = sentiment_export
                # An uncontrolled input includes the full supplied calendar.
                # It needs no construction and must not enter a shared prefix
                # cache containing OUTER rows before a checkpoint is frozen.
            scaler_key = (news_key, required)
            if scaler_cache is not None and scaler_key in scaler_cache:
                scaler = scaler_cache[scaler_key]
            else:
                scaler = _SentimentScaler.fit(prepared, exported, config, required=required)
                if scaler_cache is not None:
                    scaler_cache[scaler_key] = scaler
            exports[mode], scalers[mode] = exported, scaler
        prepared_folds[fold["fold"]] = {"prepared": prepared, "exports": exports, "scalers": scalers}
        for candidate in ablation.candidates:
            mode = _control_mode(candidate)
            control = _control_info(exports[mode], candidate) if controlled else None
            for seed in seeds:
                spec = _task_spec(metadata, prepared, scalers[mode], fold, candidate, seed, control)
                tasks.append({"candidate": candidate, "seed": seed, "fold": fold["fold"],
                              "signature": training_signature(spec), "spec": spec})
    if _runtime_cache is not None:
        _runtime_cache.update(work=work, config=config, folds=folds, prepared_folds=prepared_folds,
                              scored=scored, controlled=controlled)
    return {"metadata": metadata, "task_specs": tasks}


def _check_resume(target, plan):
    if not target.is_dir():
        raise FileNotFoundError(f"Cannot resume missing news comparison: {target}")
    saved = json.loads((target / "metadata.json").read_text())
    if _resume_metadata(saved) != _resume_metadata(plan["metadata"]):
        raise ValueError("Resume metadata does not match news data, manifest, folds, runtime or settings.")
    tasks = {(item["candidate"], item["seed"], item["fold"]): item["signature"] for item in plan["task_specs"]}
    rows = json.loads((target / "folds.json").read_text()) if (target / "folds.json").exists() else []
    keys = [(row["candidate"], row["seed"], row["fold"]) for row in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= tasks.keys():
        raise ValueError("Resume contains duplicate or unexpected news tasks.")
    # Never repair, silently retrain or overwrite a corrupt completed task.
    return [_validated_row(row, target, tasks[key]) for row, key in zip(rows, keys)]


def _exposure_report(rows, target, config, loss, ablation):
    """Descriptive ex-post common mean gross exposure; never selects a model."""
    from .exposure_comparison import normalize_positions

    results, paths = [], []
    groups = sorted({(row["fold"], row["seed"]) for row in rows})
    for fold, seed in groups:
        selected = [row for row in rows if (row["fold"], row["seed"]) == (fold, seed)]
        if {row["candidate"] for row in selected} != set(ablation.candidates):
            continue
        positions, common = {}, None
        for row in selected:
            predictions = pd.read_parquet(target / row["prediction_artifacts"]["outer"])
            predictions = predictions.sort_values(["date", "ticker"]).reset_index(drop=True)
            frame = predictions[["date", "ticker", "adj_close"]]
            if common is not None and not common.equals(frame):
                raise ValueError("Exposure report candidates are not exact matched backtest keys.")
            common = frame
            # Stored zero positions are the declared FLAT policy, not imputed
            # available signals. Availability remains in the prediction files.
            positions[row["candidate"]] = predictions.position.to_numpy(dtype=float)
        positions["buy_hold"] = np.ones(len(common))
        panel = ReturnPanel(common, group_col="ticker", execution_delay=config.execution_delay)
        adjusted, info = normalize_positions(panel, positions, "mean_min")
        for candidate, value in adjusted.items():
            results.append({"candidate": candidate, "fold": fold, "seed": seed,
                            "factor": info["scales"][candidate],
                            "target_mean_exposure": info["target_mean_exposure"],
                            "metrics": panel.metrics(value, loss, config.initial_capital)})
            daily = _daily_paths(panel, value, loss)
            daily.insert(0, "seed", seed)
            daily.insert(0, "fold", fold)
            daily.insert(0, "candidate", candidate)
            paths.append(daily)
    artifact = None
    if paths:
        path = target / "exposure-mean-min-daily.parquet"
        _atomic_parquet(pd.concat(paths, ignore_index=True), path)
        artifact = file_record(path, target)
    return {"method": "mean_min", "descriptive_only": True, "ex_post": True,
            "selection_uses_adjusted_metrics": False, "tasks": results, "daily_artifact": artifact,
            "notes": ["All candidates and delayed equal-weight buy-and-hold downscaled to one common mean gross exposure per fold/seed.",
                      "Whole-fold factors are ex post, not a deployable sizing rule.",
                      "Fresh ReturnPanel costs/liquidation; not equal net exposure, beta or volatility.",
                      "A completely flat sentiment candidate collapses the common minimum to zero."]}


def run_news_sentiment_ablation(frame, config, loss, gru_parameters, seeds, destination, *, sentiment_export,
                                ablation=NewsSentimentAblationConfig(), n_splits=3,
                                initial_train_fraction=.5, inner_val_fraction=.2,
                                gap_bars=5, embargo_bars=0, dataset_path=None,
                                resume=False, dry_run=False, progress_callback=None, _study_cache=None):
    """Train matched controls; freeze each inner checkpoint before opening outer."""
    import torch

    target = Path(destination).expanduser().resolve()
    if target.exists() and not resume:
        raise FileExistsError(f"News comparison output already exists: {target}")
    options = dict(ablation=ablation, n_splits=n_splits, initial_train_fraction=initial_train_fraction,
                   inner_val_fraction=inner_val_fraction, gap_bars=gap_bars,
                   embargo_bars=embargo_bars, dataset_path=dataset_path)
    runtime = {}
    plan = plan_run_news_sentiment_ablation(frame, config, loss, gru_parameters, seeds,
                                          sentiment_export=sentiment_export, _runtime_cache=runtime,
                                          _study_cache=_study_cache, **options)
    rows = _check_resume(target, plan) if resume else []
    if dry_run:
        return {**plan, "dry_run": True, "completed_tasks": len(rows),
                "destination": str(target), "final_holdout_opened": False}
    # Use the exact TRAIN preparation that produced the validated signatures;
    # do not recompute fitted selectors/scalers after planning.
    work, config, folds, metadata = runtime["work"], runtime["config"], runtime["folds"], plan["metadata"]
    if not resume:
        target.mkdir(parents=True)
        atomic_write_json(target / "metadata.json", metadata)
    completed = {(row["candidate"], row["seed"], row["fold"]) for row in rows}
    specs = {(item["candidate"], item["seed"], item["fold"]): item for item in plan["task_specs"]}
    training_template = GRUConfig(**gru_parameters)
    for fold in folds:
        if all((candidate, seed, fold["fold"]) in completed for candidate in ablation.candidates for seed in seeds):
            continue
        fold_frame = work.loc[_dates(work, config.date_col) <= pd.Timestamp(fold["end"])].copy()
        fold_config = replace(config, purged_split=fold["split"])
        cached = runtime["prepared_folds"][fold["fold"]]
        prepared = cached["prepared"]
        outer_prepared, outer_exports = None, {}
        train_panel = ReturnPanel(prepared.train, price_col=config.price_col, date_col=config.date_col,
                                  group_col=config.group_col, execution_delay=config.execution_delay)
        val_panel = ReturnPanel(prepared.validation, price_col=config.price_col, date_col=config.date_col,
                                group_col=config.group_col, execution_delay=config.execution_delay)
        for candidate in ablation.candidates:
            if all((candidate, seed, fold["fold"]) in completed for seed in seeds):
                continue
            mode = _control_mode(candidate)
            exported, scaler = cached["exports"][mode], cached["scalers"][mode]
            train_ds = _dataset(prepared.train, prepared.history_train, prepared, exported,
                                scaler, config, candidate)
            val_ds = _dataset(prepared.validation, prepared.history_validation, prepared, exported,
                              scaler, config, candidate)
            outer_ds = None
            for seed in seeds:
                key = candidate, seed, fold["fold"]
                if key in completed:
                    continue
                stem = f"fold-{fold['fold']}-{candidate}-seed-{seed}"
                if any(target.glob(f"{stem}*")):
                    raise FileExistsError(f"Unregistered artifacts for {stem}; refusing to overwrite. Choose a new output directory.")
                seed_torch_run(seed, training_template.deterministic, torch)
                training = replace(training_template, seed=seed, device=config.device)
                model = _make_model(next(train_ds.iter_batches(1)), candidate, config,
                                    gru_parameters, ablation, seed, torch)
                fitted = (_fit_sentiment if candidate == "sentiment" else _fit)(
                    model, train_ds, val_ds, train_panel, val_panel, loss, config, training, ablation, torch)
                model.eval()
                task = specs[key]
                spec, signature = task["spec"], task["signature"]
                model_path = target / f"{stem}.pt"
                # Save the frozen checkpoint before any outer features/labels.
                atomic_torch_save(model_path, {"schema_version": 1, "model_state": model.state_dict(),
                    "candidate": candidate, "fold": fold["fold"], "seed": seed,
                    "preprocessing": spec["preprocessing"], "training_spec": spec,
                    "task_signature": signature, "fit": fitted})
                inner_path = target / f"{stem}-inner-predictions.parquet"
                with torch.no_grad():
                    inner_positions, _, _ = _export_predictions(model, val_ds, prepared.validation, inner_path,
                        candidate=candidate, fold=fold["fold"], seed=seed, partition="inner",
                        exported=exported, config=config, ablation=ablation, torch=torch)
                # Only after this checkpoint is frozen may evaluation features
                # be opened. Reuse deterministic outer data across model seeds.
                if outer_prepared is None:
                    raw_train, raw_val, raw_outer, _, outer_columns = _prepare_splits(
                        fold_frame, fold_config, include_test=True, fill_values=prepared.fills,
                        fracdiff_transformer=prepared.fracdiff, feature_selector=prepared.selector,
                        overfitting_selector=prepared.overfitting_selector, overfitting_supervised=False)
                    if outer_columns != prepared.columns:
                        raise ValueError("Outer price features differ from the frozen train feature pool.")
                    outer = _complete(raw_outer, config, prepared.tickers)
                    outer_scaled = _scale(outer, prepared.columns, prepared.scaler)
                    history = _scale(pd.concat((raw_train, raw_val), ignore_index=True), prepared.columns, prepared.scaler)
                    outer_panel = ReturnPanel(outer, price_col=config.price_col, date_col=config.date_col,
                                             group_col=config.group_col, execution_delay=config.execution_delay)
                    outer_prepared = outer, outer_scaled, history, outer_panel
                outer, outer_scaled, history, outer_panel = outer_prepared
                if mode not in outer_exports:
                    if runtime["controlled"]:
                        from trading_system.data.fnspid_polarity import prepare_fold_news_control
                        evaluated = prepare_fold_news_control(sentiment_export, runtime["scored"], fold,
                            mode=mode, shuffle_seed=ablation.shuffle_seed, include_outer=True)
                        prefix = evaluated.frame.loc[_dates(evaluated.frame, "date") < pd.Timestamp(fold["split"].test_start)]
                        if not prefix.reset_index(drop=True).equals(exported.frame.reset_index(drop=True)):
                            raise ValueError("Outer control construction changed frozen TRAIN/INNER news.")
                        outer_exports[mode] = evaluated
                    else:
                        outer_exports[mode] = sentiment_export
                outer_export = outer_exports[mode]
                if outer_ds is None:
                    outer_ds = _dataset(outer_scaled, history, prepared, outer_export, scaler, config, candidate)
                outer_path = target / f"{stem}-outer-predictions.parquet"
                with torch.no_grad():
                    outer_positions, probs, available = _export_predictions(model, outer_ds, outer, outer_path,
                        candidate=candidate, fold=fold["fold"], seed=seed, partition="outer",
                        exported=outer_export, config=config, ablation=ablation, torch=torch)
                metrics = outer_panel.metrics(outer_positions, loss, config.initial_capital)
                row = {"candidate": candidate, "fold": fold["fold"], "seed": seed, "status": "ok",
                    "task_signature": signature, "model_artifact": model_path.name,
                    "result_artifact": f"{stem}-result.json", "fit": fitted, "score": metrics["regularized_sharpe"],
                    "inner_metrics": val_panel.metrics(inner_positions, loss, config.initial_capital), "outer_metrics": metrics,
                    "classification": _classification(outer, probs),
                    "classification_available_only": _classification(outer.loc[available], probs[available]) if available.any() else None,
                    "coverage": {name: _coverage(part, sentiment_export, config) for name, part in
                                 (("train", prepared.train), ("inner", prepared.validation), ("outer", outer))},
                    "signal_available_rows": int(available.sum()), "feature_columns": prepared.columns,
                    "effective_temporal_columns": _temporal_columns(prepared, candidate, ablation.news_protocol), "sentiment_columns": SENTIMENT_COLUMNS,
                    "sentiment_scaler": scaler.state_dict(), "purging": prepared.purging,
                    "train_dates": len(train_ds), "inner_dates": len(val_ds), "outer_dates": len(outer_ds),
                    "prediction_artifacts": {"inner": inner_path.name, "outer": outer_path.name}, "daily_path_artifacts": {},
                    "preprocessing_signature": stable_digest(spec["preprocessing"]),
                    "eligible_sessions": {**spec["eligible_sessions"], "outer": [day.isoformat() for day in _dates(outer, config.date_col).unique().sort_values()]}}
                if runtime["controlled"]:
                    row["news_control"] = _control_info(outer_export, candidate)
                for partition, panel, predicted in (("inner", val_panel, inner_positions), ("outer", outer_panel, outer_positions)):
                    daily_path = target / f"{stem}-{partition}-daily.parquet"
                    daily = _daily_paths(panel, predicted, loss)
                    for name, value in (("partition", partition), ("seed", seed), ("fold", fold["fold"]), ("variant_id", candidate)):
                        daily.insert(0, name, value)
                    _atomic_parquet(daily, daily_path)
                    row["daily_path_artifacts"][partition] = daily_path.name
                row = _nullable_metadata(row)
                atomic_write_json(target / row["result_artifact"], row)
                files = [model_path, inner_path, outer_path, target / row["result_artifact"],
                         *(target / value for value in row["daily_path_artifacts"].values())]
                row["completion_manifest"] = completion_manifest(signature, [file_record(path, target) for path in files])
                rows.append(row)
                completed.add(key)
                atomic_write_json(target / "folds.json", rows)
                if progress_callback:
                    progress_callback(row, len(specs))
                else:
                    print(f"news_cv={len(rows)}/{len(specs)} {candidate} seed={seed} fold={fold['fold']} score={row['score']:.4f}", flush=True)
    summary = [{"candidate": candidate, "complete": True,
                "mean": float(np.mean([row["score"] for row in rows if row["candidate"] == candidate])),
                "mean_net_return": float(np.mean([row["outer_metrics"]["net_return"] for row in rows if row["candidate"] == candidate])),
                "mean_gross_exposure": float(np.mean([row["outer_metrics"]["mean_abs_position"] for row in rows if row["candidate"] == candidate]))}
               for candidate in ablation.candidates]
    reference = {(row["fold"], row["seed"]): row for row in rows if row["candidate"] == "gru"}
    paired = [{"candidate": row["candidate"], "fold": row["fold"], "seed": row["seed"],
               "score_delta": row["score"] - reference[row["fold"], row["seed"]]["score"],
               "net_return_delta": row["outer_metrics"]["net_return"] - reference[row["fold"], row["seed"]]["outer_metrics"]["net_return"]}
              for row in rows if row["candidate"] != "gru" and (row["fold"], row["seed"]) in reference]
    features = {(row["fold"], row["seed"]): row for row in rows if row["candidate"] == "gru_features"}
    paired_controls = [{"candidate": row["candidate"], "reference": "gru_features", "fold": row["fold"], "seed": row["seed"],
        "score_delta": row["score"] - features[row["fold"], row["seed"]]["score"],
        "net_return_delta": row["outer_metrics"]["net_return"] - features[row["fold"], row["seed"]]["outer_metrics"]["net_return"]}
        for row in rows if row["candidate"] in POLARITY_CANDIDATES[3:] and (row["fold"], row["seed"]) in features]
    report = _nullable_metadata({"metadata": metadata, "folds": rows, "summary": summary,
        "paired_vs_gru": paired, "selected": max(summary, key=lambda item: item["mean"])["candidate"],
        "final_test": [], "exposure_controlled": _exposure_report(rows, target, config, loss, ablation)})
    if runtime["controlled"]:
        report["paired_vs_original_features"] = paired_controls
    atomic_write_json(target / "report.json", report)
    return report


__all__ = ["CANDIDATES", "ALL_CANDIDATES", "POLARITY_CANDIDATES", "POLARITY_STATISTICS", "NewsSentimentAblationConfig", "plan_run_news_sentiment_ablation", "run_news_sentiment_ablation"]
