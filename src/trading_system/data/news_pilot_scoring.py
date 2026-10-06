"""Offline, provenance-preserving FinBERT scoring for the observed news pilot.

The optional news_sentiment package owns inference validation and aggregation.
Collector observations are never promoted into historical coverage evidence.
Macro associations are exported separately, not replicated across equities.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from .news_sentiment import (
    SENTIMENT_COLUMNS,
    build_news_decision_points,
    load_news_sentiment_export,
)


ARTICLE_COLUMNS = (
    "article_id", "url", "source", "title", "summary", "published_at",
    "collected_at", "available_at", "availability_kind", "availability_reference",
    "raw_reference", "raw_sha256", "content_kind",
)
ASSOCIATION_COLUMNS = ("article_id", "scope_type", "scope")
ASSOCIATION_EVIDENCE_COLUMNS = ("scope_available_at", "scope_raw_reference", "scope_raw_sha256")
PREDICTION_COLUMNS = (
    "text", "label", "confidence", "p_negative", "p_neutral", "p_positive",
    "sentiment_score",
)
SCORE_COLUMNS = PREDICTION_COLUMNS[1:]
LIBRARY_REVISION = "15424a2c9fd086f4af1740a22c4a9cd032981e40"


@dataclass(frozen=True)
class FinBERTCheckpoint:
    """Immutable model identity; the CLI additionally verifies local weights."""

    repository: str
    revision: str
    weights_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not self.repository.strip():
            raise ValueError("Model repository must be non-empty.")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Model revision must be an immutable lowercase 40-hex commit SHA.")
        if not isinstance(self.weights_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.weights_sha256):
            raise ValueError("Model weights SHA256 must be a lowercase 64-hex digest.")

    @property
    def identifier(self) -> str:
        return f"{self.repository}@{self.revision};weights_sha256={self.weights_sha256}"


@dataclass
class ScoredNewsPilot:
    articles: pd.DataFrame
    company: pd.DataFrame
    macro: pd.DataFrame
    output_dir: Path
    manifest: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_frame(path: Path, frame: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        manifest = {
            **metadata, "schema_version": "1.0", "feature_rows": len(frame),
            "feature_columns": list(frame), "parquet_sha256": file_sha256(temporary),
        }
        os.replace(temporary, path)
        _write_json(path.with_suffix(".manifest.json"), manifest)
        return manifest
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _aware(values: pd.Series, field: str, *, nullable: bool = False) -> pd.Series:
    timestamps = []
    for value in values:
        if nullable and pd.isna(value):
            timestamps.append(pd.NaT)
            continue
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError(f"{field} requires explicit timezone-aware timestamps.")
        timestamps.append(timestamp.tz_convert("UTC"))
    return pd.Series(pd.to_datetime(timestamps, utc=True), index=values.index)


def _validate_inputs(articles: pd.DataFrame, associations: pd.DataFrame) -> pd.DataFrame:
    for name, frame, columns in (
        ("articles", articles, ARTICLE_COLUMNS),
        ("associations", associations, (*ASSOCIATION_COLUMNS, *ASSOCIATION_EVIDENCE_COLUMNS)),
    ):
        missing = sorted(set(columns) - set(frame))
        if missing or frame.columns.has_duplicates:
            raise ValueError(f"{name} has missing or duplicate columns: {missing}")
    for column in ("article_id", "source", "availability_reference", "raw_reference"):
        if not articles[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Articles require non-empty {column}.")
    if articles["article_id"].duplicated().any():
        raise ValueError("Articles must have unique article_id values.")
    if not articles["raw_sha256"].map(
        lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    ).all():
        raise ValueError("Articles require raw_sha256 evidence digests.")
    if not articles["availability_kind"].eq("collector_first_seen").all():
        raise ValueError("This pilot accepts collector_first_seen, not inferred historical availability.")
    if not articles["content_kind"].eq("title_summary").all():
        raise ValueError("This pilot scores title_summary, not full article bodies.")
    if not associations["scope_type"].isin(("company", "macro")).all():
        raise ValueError("Associations must have company or macro scope_type.")
    if not associations["scope"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Associations require non-empty scopes.")
    if associations.duplicated(list(ASSOCIATION_COLUMNS)).any():
        raise ValueError("Duplicate article/scope associations are not allowed.")
    if not associations["article_id"].isin(articles["article_id"]).all():
        raise ValueError("Associations reference unknown articles.")
    reserved = set(PREDICTION_COLUMNS) | {"news_id", "checkpoint", "content_sha256", "collector_availability_kind"}
    if reserved.intersection(articles) or reserved.intersection(associations):
        raise ValueError("Collector inputs must not contain FinBERT prediction/reserved columns.")
    if (set(articles) & set(associations)) - {"article_id"}:
        raise ValueError("Article and association columns overlap beyond article_id.")
    checked = articles.copy().reset_index(drop=True)
    for column in ("collected_at", "available_at"):
        checked[column] = _aware(checked[column], column)
    checked["published_at"] = _aware(checked["published_at"], "published_at", nullable=True)
    if checked["available_at"].gt(checked["collected_at"]).any():
        raise ValueError("First observed availability cannot follow collection time.")
    texts = []
    for title, summary in checked[["title", "summary"]].itertuples(index=False, name=None):
        parts = []
        for value in (title, summary):
            if pd.isna(value):
                value = ""
            if not isinstance(value, str):
                raise ValueError("Title and summary must be strings or missing.")
            normalized = " ".join(value.split())
            if normalized and normalized not in parts:
                parts.append(normalized)
        text = "\n".join(parts)
        if not text:
            raise ValueError("Every article needs a non-empty title or summary.")
        texts.append(text)
    checked["text"] = texts
    checked["content_sha256"] = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    return checked


def _predict(analyzer: Any, texts: list[str]) -> pd.DataFrame:
    # Public predict() at the pinned revision calls the library's _predict_many.
    # A predict_many/callable adapter is useful for offline tests and newer APIs.
    method = getattr(analyzer, "predict", None) or getattr(analyzer, "predict_many", None)
    if method is None and callable(analyzer):
        method = analyzer
    if not callable(method):
        raise TypeError("Analyzer must provide predict/predict_many or be callable.")
    result = method(texts)
    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame.from_records(
            [item.to_record() if hasattr(item, "to_record") else item for item in result]
        )
    if len(result) != len(texts):
        raise ValueError("Analyzer returned the wrong prediction count.")
    from news_sentiment.stats import validate_prediction_frame
    validate_prediction_frame(result)
    if result["text"].tolist() != texts:
        raise ValueError("Analyzer prediction text/order changed.")
    return result.loc[:, PREDICTION_COLUMNS].reset_index(drop=True)


def _resume_predictions(
    output_dir: Path, inputs: dict[str, str], checkpoint: str,
    settings: dict[str, Any], articles: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str]:
    manifest_path = output_dir / "scoring.manifest.json"
    target = output_dir / "scored_articles.parquet"
    if not manifest_path.exists() or not target.exists():
        return None, "missing_cache"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_identifiers") != inputs:
            return None, "changed_inputs"
        if manifest.get("checkpoint") != checkpoint or manifest.get("inference_settings") != settings:
            return None, "changed_checkpoint_or_settings"
        if manifest.get("artifacts", {}).get(target.name, {}).get("sha256") != file_sha256(target):
            return None, "artifact_checksum_mismatch"
        cached = pd.read_parquet(target)
        if cached["article_id"].tolist() != articles["article_id"].tolist() or (
            cached["content_sha256"].tolist() != articles["content_sha256"].tolist()
        ) or cached["text"].tolist() != articles["text"].tolist():
            return None, "changed_content"
        if not cached["checkpoint"].eq(checkpoint).all():
            return None, "changed_checkpoint"
        from news_sentiment.stats import validate_prediction_frame
        if not cached.empty:
            validate_prediction_frame(cached)
        return cached.loc[:, SCORE_COLUMNS], "valid_cache"
    except (OSError, ValueError, KeyError, TypeError):
        return None, "invalid_cache"


def _verify_collection(input_dir: Path, articles: pd.DataFrame, associations: pd.DataFrame) -> None:
    """Verify the collector's tables and physical first-observation evidence."""
    manifest_path = input_dir / "collection.manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Scoring requires the collector manifest and raw evidence.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported collection manifest schema.")
    for name in ("articles.parquet", "associations.parquet"):
        entry = manifest.get("files", {}).get(name, {})
        if entry.get("sha256") != file_sha256(input_dir / name):
            raise ValueError(f"Collection checksum mismatch: {name}.")
    requests = [item for record in manifest.get("requests", [])
                for item in (record, *record.get("attempts", []))]
    observations = {item["raw_reference"]: item for item in requests
                    if item.get("raw_reference") and item.get("finished_at")}
    root = input_dir.resolve()
    evidence_rows = pd.concat((
        articles[["raw_reference", "raw_sha256", "available_at"]],
        associations[list(ASSOCIATION_EVIDENCE_COLUMNS)].rename(columns={
            "scope_raw_reference": "raw_reference", "scope_raw_sha256": "raw_sha256",
            "scope_available_at": "available_at",
        }),
    ), ignore_index=True)
    evidence_rows["available_at"] = _aware(evidence_rows["available_at"], "scope_available_at")
    for reference, group in evidence_rows.groupby("raw_reference", sort=False):
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("Raw evidence reference must be non-empty.")
        relative = Path(reference)
        raw = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not raw.is_relative_to(root):
            raise ValueError("Unsafe raw evidence reference.")
        if not raw.is_file() or not group["raw_sha256"].eq(file_sha256(raw)).all():
            raise ValueError("Raw evidence checksum mismatch.")
        evidence = observations.get(reference)
        if evidence is None or evidence.get("raw_sha256") != file_sha256(raw):
            raise ValueError("Raw evidence lacks a matching collector observation.")
        observed = pd.Timestamp(evidence["finished_at"])
        if observed.tzinfo is None or not group["available_at"].eq(observed).all():
            raise ValueError("Availability does not match first collector observation.")


def score_news_pilot(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    checkpoint: FinBERTCheckpoint,
    analyzer: Any = None,
    analyzer_factory: Callable[[], Any] | None = None,
    resume: bool = False,
    inference_settings: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ScoredNewsPilot:
    """Score unique normalized title/summary content once, then route scopes.

    Resume is only allowed when input files, content, checkpoint, settings and
    the scored-artifact checksum agree. Changed or invalid runs are not overwritten.
    No API response or collection manifest is converted into coverage evidence.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError("Scoring output already exists; use --resume or a new version directory.")
    article_path, association_path = input_dir / "articles.parquet", input_dir / "associations.parquet"
    inputs = {"articles_sha256": file_sha256(article_path), "associations_sha256": file_sha256(association_path)}
    collection_manifest = input_dir / "collection.manifest.json"
    if collection_manifest.exists():
        inputs["collection_manifest_sha256"] = file_sha256(collection_manifest)
    associations = pd.read_parquet(association_path)
    articles = _validate_inputs(pd.read_parquet(article_path), associations)
    _verify_collection(input_dir, articles, associations)
    if articles.empty:
        raise ValueError("Collection contains no articles; inspect its incomplete/vendor-access status.")
    settings = dict(inference_settings or {})
    scores, cache_status = _resume_predictions(output_dir, inputs, checkpoint.identifier, settings, articles) if resume else (None, "disabled")
    if resume and scores is None:
        raise ValueError(f"Scoring resume refused: {cache_status}. Use a new version directory.")
    unique = articles.loc[:, ["content_sha256", "text"]].drop_duplicates("content_sha256")
    scored_texts = 0
    if scores is None:
        if not unique.empty:
            if analyzer is None:
                if analyzer_factory is None:
                    raise ValueError("An analyzer or analyzer_factory is required for uncached articles.")
                if progress:
                    progress("Loading verified local FinBERT checkpoint (offline).")
                analyzer = analyzer_factory()
            texts = unique["text"].tolist()
            batch_size = settings.get("batch_size", len(texts))
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError("Inference batch_size must be a positive integer.")
            batches = []
            for start in range(0, len(texts), batch_size):
                batches.append(_predict(analyzer, texts[start:start + batch_size]))
                if progress:
                    progress(f"FinBERT scored {min(start + batch_size, len(texts))}/{len(texts)} unique title/summary texts.")
            predictions = pd.concat(batches, ignore_index=True)
            predictions.index = unique["content_sha256"].to_numpy()
            scores = predictions.loc[articles["content_sha256"], SCORE_COLUMNS].reset_index(drop=True)
            scored_texts = len(unique)
        else:
            scores = pd.DataFrame({name: pd.Series(dtype="object" if name == "label" else "float64") for name in SCORE_COLUMNS})
    articles = pd.concat([articles, scores.reset_index(drop=True)], axis=1)
    articles["checkpoint"] = checkpoint.identifier
    articles["news_id"] = articles["article_id"]
    articles["collector_availability_kind"] = articles["availability_kind"]
    articles["availability_kind"] = "pipeline_observed"
    routed = associations.merge(articles, on="article_id", how="left", validate="many_to_one")
    # Knowing a text is not equivalent to knowing a later vendor association.
    # A query crossing midnight must not backdate a newly observed ticker/topic.
    routed["article_available_at"] = routed["available_at"]
    routed["scope_available_at"] = _aware(routed["scope_available_at"], "scope_available_at")
    routed["available_at"] = routed[["article_available_at", "scope_available_at"]].max(axis=1)
    routed["availability_reference"] = routed["scope_raw_reference"]
    company = routed.loc[routed["scope_type"].eq("company")].copy().reset_index(drop=True)
    company["ticker"] = company["scope"]
    macro = routed.loc[routed["scope_type"].eq("macro")].copy().reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, frame, domain in (
        ("scored_articles.parquet", articles, "article"),
        ("scored_news_company.parquet", company, "company"),
        ("scored_news_macro.parquet", macro, "macro"),
    ):
        metadata = _write_frame(output_dir / name, frame, {
            "domain": domain, "checkpoint": checkpoint.identifier,
            "input_identifiers": inputs, "content_kind": "title_summary",
            "availability_mapping": {"collector_first_seen": "pipeline_observed"},
        })
        artifacts[name] = {"sha256": metadata["parquet_sha256"], "rows": len(frame), "domain": domain}
    manifest = {
        "schema_version": "1.0", "checkpoint": checkpoint.identifier,
        "model": {"repository": checkpoint.repository, "revision": checkpoint.revision, "weights_sha256": checkpoint.weights_sha256},
        "library_revision": LIBRARY_REVISION, "input_identifiers": inputs,
        "inference_settings": settings, "content_kind": "title_summary",
        "unique_texts": len(unique), "newly_scored_texts": scored_texts,
        "resume_cache_status": cache_status, "artifacts": artifacts,
        "coverage_note": "Collector observations establish availability only, never exhaustive historical coverage.",
    }
    _write_json(output_dir / "scoring.manifest.json", manifest)
    return ScoredNewsPilot(articles, company, macro, output_dir, manifest)


def verify_local_weights(model_dir: str | Path, expected_sha256: str) -> Path:
    directory = Path(model_dir).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("--model-dir must be an existing local model directory.")
    candidates = [directory / name for name in ("model.safetensors", "pytorch_model.bin") if (directory / name).is_file()]
    if len(candidates) != 1:
        raise ValueError("Local model directory must contain exactly one model.safetensors or pytorch_model.bin.")
    if file_sha256(candidates[0]) != expected_sha256:
        raise ValueError("Local model weights do not match --model-weights-sha256.")
    return candidates[0]


def load_local_finbert(
    model_dir: str | Path,
    *,
    checkpoint: FinBERTCheckpoint,
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 512,
) -> Any:
    """Load locally only, staging missing legacy BERT metadata without edits.

    The pinned library's remote-name workaround does not apply to local paths.
    Copy metadata to a temporary directory and provide model_type/tokenizer_class
    there; hard-link verified weights (copy fallback). Original files stay untouched.
    Both original config and the documented staging policy are recorded by CLI.
    """
    directory = Path(model_dir).resolve(strict=True)
    weights = verify_local_weights(directory, checkpoint.weights_sha256)
    from news_sentiment import SentimentAnalyzer
    # Disable hub access even if a locally supplied metadata file references it.
    previous = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ.update({key: "1" for key in previous})
    try:
        with tempfile.TemporaryDirectory(prefix="finbert-local-", dir=directory.parent) as temporary:
            stage = Path(temporary)
            for source in directory.iterdir():
                if source.is_file() and source.suffix in (".json", ".txt", ".model"):
                    shutil.copy2(source, stage / source.name)
            try:
                os.link(weights, stage / weights.name)
            except OSError:
                shutil.copy2(weights, stage / weights.name)
            config_path = stage / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if "model_type" not in config:
                if not any(name.startswith("Bert") for name in config.get("architectures", [])):
                    raise ValueError("Missing model_type is only repaired for an explicitly declared BERT architecture.")
                config["model_type"] = "bert"
                _write_json(config_path, config)
            tokenizer_path = stage / "tokenizer_config.json"
            if config.get("model_type") == "bert":
                tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8")) if tokenizer_path.exists() else {}
                tokenizer.setdefault("tokenizer_class", "BertTokenizer")
                _write_json(tokenizer_path, tokenizer)
            return SentimentAnalyzer(str(stage), device=device, batch_size=batch_size, max_length=max_length)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _sources(coverage: pd.DataFrame | None, required_sources: Sequence[str] | None) -> list[str] | None:
    if (coverage is None) != (required_sources is None):
        raise ValueError("Coverage and explicit required_sources must be supplied together.")
    if coverage is None:
        return None
    if isinstance(required_sources, str):
        raise TypeError("required_sources must be a sequence, not a string.")
    sources = list(required_sources or ())
    if not sources or len(sources) != len(set(sources)) or any(not isinstance(value, str) or not value.strip() for value in sources):
        raise ValueError("required_sources must be unique non-empty identifiers.")
    from news_sentiment.coverage import validate_coverage
    validate_coverage(coverage)
    return sources


def export_news_pilot(
    pilot: ScoredNewsPilot,
    market: pd.DataFrame,
    *,
    tickers: Sequence[str],
    coverage: pd.DataFrame | None = None,
    required_sources: Sequence[str] | None = None,
    macro_coverage: pd.DataFrame | None = None,
    macro_required_sources: Sequence[str] | None = None,
    input_identifiers: dict[str, str] | None = None,
    lookback: str = "24h",
    short_lookback: str = "6h",
    half_life: str = "6h",
) -> dict[str, Path]:
    """Export strict-midnight company features and an independent macro panel.

    Company output uses the existing audited adapter (23 feature channels).
    Macro is date-only with a separate 23-channel block and mask per topic plus
    a deduplicated global block. Internal aggregator scope IDs never escape into
    public Parquet artifacts. Macro coverage must be supplied separately; its
    ticker column may identify original topic names, "global", or null.
    """
    from news_sentiment import export_sentiment_features
    sources = _sources(coverage, required_sources)
    macro_sources = _sources(macro_coverage, macro_required_sources)
    points = build_news_decision_points(market, tickers=tickers)
    identifiers = {**pilot.manifest["input_identifiers"], **(input_identifiers or {})}
    checkpoint = pilot.manifest["checkpoint"]
    windows = {"lookback": lookback, "short_lookback": short_lookback, "half_life": half_life, "include_at_cutoff": False}
    company_path = pilot.output_dir / "company_sentiment.parquet"
    export_sentiment_features(
        pilot.company.loc[pilot.company["ticker"].isin(tickers)], points, company_path,
        checkpoint=checkpoint, input_identifiers={**identifiers, "domain": "company"},
        coverage=coverage, required_sources=sources, **windows,
    )
    company = load_news_sentiment_export(company_path)
    _write_frame(pilot.output_dir / "company_panel.parquet", company.frame, {
        "domain": "company", "checkpoint": checkpoint,
        "input_identifiers": {**identifiers, "company_export_sha256": file_sha256(company_path)},
        "feature_channels": list(SENTIMENT_COLUMNS), "aggregation": company.manifest["aggregation"],
    })
    topics = sorted(pilot.macro["scope"].unique().tolist())
    # These identifiers exist only inside the temporary library export.
    groups = [("global", "macro_global", "__macro_global__")]
    for topic in topics:
        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_") or "topic"
        prefix = f"macro_{slug}_{hashlib.sha256(topic.encode()).hexdigest()[:8]}"
        groups.append((topic, prefix, f"__macro_topic__{topic}"))
    chunks = [pilot.macro.drop_duplicates("article_id").assign(ticker=groups[0][2])]
    for topic, _, internal_id in groups[1:]:
        chunks.append(pilot.macro.loc[pilot.macro["scope"].eq(topic)].assign(ticker=internal_id))
    internal_news = pd.concat(chunks, ignore_index=True)
    dates = points["decision_at"].drop_duplicates().sort_values().tolist()
    macro_points = pd.DataFrame([
        {"ticker": internal_id, "decision_at": date}
        for date in dates for _, _, internal_id in groups
    ])
    internal_coverage = None
    if macro_coverage is not None:
        internal_coverage = macro_coverage.copy()
        route = {topic: internal_id for topic, _, internal_id in groups}
        if not internal_coverage["ticker"].dropna().isin(route).all():
            raise ValueError("Macro coverage ticker must identify a macro topic, global, or be missing.")
        internal_coverage["ticker"] = internal_coverage["ticker"].map(route).where(internal_coverage["ticker"].notna(), None)
    with tempfile.TemporaryDirectory(prefix="macro-aggregation-") as temporary:
        internal_path = Path(temporary) / "features.parquet"
        export_sentiment_features(
            internal_news, macro_points, internal_path,
            checkpoint=checkpoint, input_identifiers={**identifiers, "domain": "macro"},
            coverage=internal_coverage, required_sources=macro_sources, **windows,
        )
        macro_ready = load_news_sentiment_export(internal_path)
        macro_frame = pd.DataFrame({"date": pd.to_datetime(dates, utc=True)})
        group_metadata = []
        for topic, prefix, internal_id in groups:
            frame = macro_ready.frame.loc[macro_ready.frame["ticker"].eq(internal_id)].reset_index(drop=True)
            for name in SENTIMENT_COLUMNS:
                macro_frame[f"{prefix}__{name}"] = frame[name]
            for name in ("source_available", "available_at", "coverage_status"):
                macro_frame[f"{prefix}__{name}"] = frame[name]
            group_metadata.append({
                "topic": topic, "prefix": prefix,
                "feature_columns": [f"{prefix}__{name}" for name in SENTIMENT_COLUMNS],
                "mask_column": f"{prefix}__source_available",
                "available_at_column": f"{prefix}__available_at",
                "coverage_status_column": f"{prefix}__coverage_status",
            })
        macro_path = pilot.output_dir / "macro_sentiment.parquet"
        _write_frame(macro_path, macro_frame, {
            "domain": "macro", "checkpoint": checkpoint,
            "input_identifiers": {**identifiers, "domain": "macro"},
            "aggregation": macro_ready.manifest["aggregation"],
            "inputs": {key: value for key, value in macro_ready.manifest["inputs"].items() if key != "decision_points"},
            "groups": group_metadata,
            "routing": "Independent date-only global/topic panel; never replicated across company tickers.",
        })
    return {
        "company_sentiment": company_path,
        "company_panel": pilot.output_dir / "company_panel.parquet",
        "macro_sentiment": macro_path,
    }


__all__ = [
    "FinBERTCheckpoint", "ScoredNewsPilot", "score_news_pilot", "export_news_pilot",
    "load_local_finbert", "verify_local_weights", "file_sha256",
]
