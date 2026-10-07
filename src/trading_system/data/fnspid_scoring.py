"""Offline, immutable and shard-resumable FinBERT scoring for FNSPID.

Only the locally verified checkpoint is loaded. Imported publication dates are
preserved as source metadata; this module never creates availability evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .news_collection import _output_lock
from .news_pilot_scoring import (
    FinBERTCheckpoint,
    LIBRARY_REVISION,
    PREDICTION_COLUMNS,
    SCORE_COLUMNS,
    _write_json,
    file_sha256,
    load_local_finbert,
    verify_local_weights,
)


ARTICLE_COLUMNS = (
    "news_id", "text_hash", "text", "ticker", "published_at", "collected_at",
    "source", "url", "date_raw", "association_kind", "content_kind", "source_files",
)
TIMING_FLAGS = ("version_timing_uncertain", "publication_timing_uncertain")
STAGING_POLICY = (
    "copy metadata; add missing BERT model_type/tokenizer_class; "
    "hard-link verified weights or copy fallback"
)


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _validate_articles(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = sorted(set(ARTICLE_COLUMNS) - set(frame))
    if missing or frame.columns.has_duplicates:
        raise ValueError(f"FNSPID articles have missing or duplicate columns: {missing}.")
    if set(frame).intersection((*SCORE_COLUMNS, "checkpoint")):
        raise ValueError("FNSPID input must not already contain prediction columns.")
    for name in ("news_id", "ticker", "text", "text_hash"):
        if not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"FNSPID articles require non-empty {name} strings.")
    if frame.duplicated(["news_id", "ticker"]).any():
        raise ValueError("Duplicate FNSPID news_id/ticker associations are not allowed.")
    if not frame["text_hash"].map(lambda value: re.fullmatch(r"[0-9a-f]{64}", value) is not None).all():
        raise ValueError("FNSPID text_hash must be a lowercase SHA256 digest.")
    hashes = frame["text"].map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    if not frame["text_hash"].eq(hashes).all():
        raise ValueError("FNSPID text_hash does not match its text.")
    if frame.groupby("news_id")["text_hash"].nunique().gt(1).any():
        raise ValueError("A FNSPID news_id must identify exactly one text version.")
    rejected = pd.Series(False, index=frame.index)
    counts = {}
    for name in TIMING_FLAGS:
        if name in frame:
            if not pd.api.types.is_bool_dtype(frame[name]) or frame[name].isna().any():
                raise ValueError(f"{name} must contain non-null boolean values.")
            values = frame[name]
        else:
            values = pd.Series(False, index=frame.index)
        counts[name] = int(values.sum())
        rejected |= values
    counts["excluded_associations"] = int(rejected.sum())
    return frame.loc[~rejected].reset_index(drop=True).copy(), counts


def _versions() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "transformers", "news-sentiment-feature-engineering"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _resolved_device(device: str, *, injected: bool) -> str:
    if device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'.")
    try:
        import torch
    except ModuleNotFoundError as error:
        if injected and device in ("auto", "cpu"):
            return "cpu"
        raise RuntimeError('FinBERT requires the optional sentiment/PyTorch dependencies.') from error
    from trading_system.models.neural.trainer import resolve_device
    return str(resolve_device(device, torch))


def _model_metadata(model_dir: str | Path | None, checkpoint: FinBERTCheckpoint) -> dict[str, Any]:
    if model_dir is None:
        return {"loader": "injected_analyzer", "metadata_sha256": {}}
    directory = Path(model_dir).resolve(strict=True)
    weights = verify_local_weights(directory, checkpoint.weights_sha256)
    for name in ("config.json", "vocab.txt"):
        if not (directory / name).is_file():
            raise ValueError(f"Verified local FinBERT requires {name}.")
    return {
        "loader": "load_local_finbert", "weights_file": weights.name,
        "metadata_sha256": {
            path.name: file_sha256(path) for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix in (".json", ".txt", ".model")
        },
        "local_metadata_staging": STAGING_POLICY,
    }


def _validate_predictions(frame: pd.DataFrame, texts: list[str]) -> pd.DataFrame:
    if frame.columns.has_duplicates or not set(PREDICTION_COLUMNS).issubset(frame):
        raise ValueError("Analyzer returned missing or duplicate prediction columns.")
    if len(frame) != len(texts) or frame["text"].tolist() != texts:
        raise ValueError("Analyzer changed prediction text/order/count.")
    result = frame.loc[:, PREDICTION_COLUMNS].reset_index(drop=True).copy()
    for name in ("p_negative", "p_neutral", "p_positive", "confidence", "sentiment_score"):
        if not pd.api.types.is_numeric_dtype(result[name]) or pd.api.types.is_bool_dtype(result[name]):
            raise ValueError(f"Analyzer {name} must contain numeric values.")
    probabilities = result[["p_negative", "p_neutral", "p_positive"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Analyzer probabilities must be finite values in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Analyzer probabilities must sum to one.")
    confidence = result["confidence"].to_numpy(dtype=float)
    score = result["sentiment_score"].to_numpy(dtype=float)
    if not np.isfinite(confidence).all() or not np.allclose(confidence, probabilities.max(axis=1), rtol=0, atol=1e-6):
        raise ValueError("Analyzer confidence must equal the largest probability.")
    if not np.isfinite(score).all() or not np.allclose(score, probabilities[:, 2] - probabilities[:, 0], rtol=0, atol=1e-6):
        raise ValueError("Analyzer sentiment_score must equal p_positive - p_negative.")
    labels = np.asarray(["negative", "neutral", "positive"])[probabilities.argmax(axis=1)]
    if not result["label"].eq(labels).all():
        raise ValueError("Analyzer label does not match its probabilities.")
    return result


def _predict(analyzer: Any, texts: list[str]) -> pd.DataFrame:
    method = getattr(analyzer, "predict", None) or getattr(analyzer, "predict_many", None)
    if method is None and callable(analyzer):
        method = analyzer
    if not callable(method):
        raise TypeError("Analyzer must provide predict/predict_many or be callable.")
    output = method(texts)
    if not isinstance(output, pd.DataFrame):
        output = pd.DataFrame.from_records([
            item.to_record() if hasattr(item, "to_record") else item for item in output
        ])
    return _validate_predictions(output, texts)


def _refuse(message: str) -> None:
    raise ValueError(f"FNSPID scoring resume refused: {message}. Use a new destination or restore the registered artifacts.")


def _load_manifest(path: Path, signature: dict[str, Any]) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _refuse(f"missing or invalid manifest ({error})")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0":
        _refuse("unsupported manifest schema")
    if manifest.get("signature") != signature:
        _refuse("changed input, checkpoint, inference settings, runtime or model metadata")
    if manifest.get("state") not in ("running", "complete"):
        _refuse("invalid scoring state")
    if not isinstance(manifest.get("shards"), list) or not isinstance(manifest.get("artifacts"), dict):
        _refuse("invalid artifact registry")
    return manifest


def _registered_files(destination: Path, manifest: dict[str, Any]) -> None:
    if set(manifest["artifacts"]) - {"scored_company.parquet"}:
        _refuse("unexpected final artifact registry")
    required = {"scoring.manifest.json"}
    entries = [*manifest["shards"], *manifest["artifacts"].values()]
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("status") != "complete":
            _refuse("interrupted prepared artifact commit; no existing data will be overwritten")
        path = entry.get("path")
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            _refuse("invalid artifact path")
        required.add(path)
    found = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    # The OS coordination lock is not a durable scoring artifact. A copied
    # read-only reuse cache may legitimately omit it; all registered data stays
    # mandatory and every other unregistered file remains forbidden.
    if found - {".collection.lock"} != required:
        _refuse("unregistered or missing files")
    directories = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_dir()}
    if directories - {"shards"}:
        _refuse("unregistered directories")


def _load_shards(
    destination: Path, manifest: dict[str, Any], unique: pd.DataFrame,
    shard_size: int, checkpoint: FinBERTCheckpoint,
) -> list[pd.DataFrame]:
    frames = []
    for index, entry in enumerate(manifest["shards"]):
        start, end = index * shard_size, min((index + 1) * shard_size, len(unique))
        expected_path = f"shards/part-{index:06d}.parquet"
        if start >= end or entry.get("path") != expected_path or entry.get("start") != start or entry.get("end") != end:
            _refuse("invalid or non-contiguous shard registry")
        path = destination / expected_path
        if entry.get("sha256") != file_sha256(path) or entry.get("rows") != end - start:
            _refuse(f"shard checksum/row mismatch: {expected_path}")
        try:
            frame = pd.read_parquet(path)
            if list(frame) != ["text_hash", *PREDICTION_COLUMNS, "checkpoint"]:
                _refuse(f"changed shard columns: {expected_path}")
            _validate_predictions(frame, unique.iloc[start:end]["text"].tolist())
            if frame["text_hash"].tolist() != unique.iloc[start:end]["text_hash"].tolist() or not frame["checkpoint"].eq(checkpoint.identifier).all():
                _refuse(f"changed shard content/checkpoint: {expected_path}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            _refuse(f"invalid shard {expected_path} ({error})")
        frames.append(frame)
    return frames


def _commit_frame(
    frame: pd.DataFrame, destination: Path, manifest_path: Path,
    manifest: dict[str, Any], entry: dict[str, Any], *, shard: bool,
) -> None:
    """Prepare a durable manifest before the atomic artifact commit.

    A crash during the commit leaves a prepared entry. Resume refuses it rather
    than overwriting an already committed artifact or guessing its completion.
    """
    target = destination / entry["path"]
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing scoring artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=".parquet", dir=target.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        entry.update({"sha256": file_sha256(temporary), "rows": len(frame), "status": "prepared"})
        if shard:
            manifest["shards"].append(entry)
        else:
            manifest["artifacts"][target.name] = entry
        _write_json(manifest_path, manifest)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite concurrently created scoring artifact: {target}")
        # Both paths share a filesystem. Creating the final hard link is an
        # atomic no-overwrite commit, including when another process races us.
        os.link(temporary, target)
        os.unlink(temporary)
        entry["status"] = "complete"
        _write_json(manifest_path, manifest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _route(articles: pd.DataFrame, shards: list[pd.DataFrame], checkpoint: FinBERTCheckpoint) -> pd.DataFrame:
    if shards:
        scores = pd.concat(shards, ignore_index=True)
        result = articles.merge(scores[["text_hash", *SCORE_COLUMNS]], on="text_hash", how="left", sort=False, validate="many_to_one")
    else:
        result = articles.copy()
        for name in SCORE_COLUMNS:
            result[name] = pd.Series(dtype="object" if name == "label" else "float64", index=result.index)
    result["checkpoint"] = checkpoint.identifier
    return result


def _reuse_refuse(message: str) -> None:
    raise ValueError(f"FNSPID scoring reuse refused: {message}.")


def _load_reuse_sources(
    sources: Sequence[str | Path], signature: dict[str, Any],
    checkpoint: FinBERTCheckpoint, destination: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Read and verify completed caches before creating any target artifacts.

    The final company route supplies the original eligible text inventory. Its
    hashes, associations and scores must agree with every registered shard. A
    source's article file need not remain alongside its immutable scoring cache.
    """
    frames: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    try:
        paths = sorted({Path(source).resolve(strict=True) for source in sources}, key=str)
    except (OSError, TypeError) as error:
        _reuse_refuse(f"unavailable cache directory ({error})")
    for directory in paths:
        if directory == destination.resolve():
            _reuse_refuse("a scoring destination cannot reuse itself")
        manifest_path = directory / "scoring.manifest.json"
        try:
            manifest_sha256 = file_sha256(manifest_path)
            source = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(source, dict) or not isinstance(source.get("signature"), dict):
                _reuse_refuse(f"invalid cache manifest: {directory}")
            source_signature = source["signature"]
            source = _load_manifest(manifest_path, source_signature)
            if source["state"] != "complete":
                _reuse_refuse(f"cache is not complete: {directory}")
            for field in ("checkpoint", "model_metadata", "library_revision", "runtime_versions", "timing_policy"):
                if source_signature.get(field) != signature[field]:
                    _reuse_refuse(f"incompatible cache {field}: {directory}")
            source_settings = source_signature.get("inference_settings")
            if not isinstance(source_settings, dict):
                _reuse_refuse(f"invalid cache inference settings: {directory}")
            # Shard boundaries and requested auto/cpu aliases do not change
            # inference. Device actually used, batch and tokenization do.
            for field in ("resolved_device", "batch_size", "max_length"):
                if source_settings.get(field) != signature["inference_settings"][field]:
                    _reuse_refuse(f"incompatible cache inference setting {field}: {directory}")
            source_shard_size = _positive_integer(source_settings.get("shard_size"), "cache shard_size")
            for field in ("checkpoint", "inference_settings", "library_revision", "runtime_versions"):
                if source.get(field) != source_signature.get(field):
                    _reuse_refuse(f"inconsistent cache manifest {field}: {directory}")
            expected_model = {
                "repository": checkpoint.repository, "revision": checkpoint.revision,
                "weights_sha256": checkpoint.weights_sha256, **signature["model_metadata"],
            }
            if source.get("model") != expected_model:
                _reuse_refuse(f"inconsistent cache model identity: {directory}")
            input_sha256 = source_signature.get("articles_sha256")
            if not isinstance(input_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
                _reuse_refuse(f"invalid cache input fingerprint: {directory}")
            if source.get("input_identifiers") != {"articles_sha256": input_sha256}:
                _reuse_refuse(f"inconsistent cache input fingerprint: {directory}")
            _registered_files(directory, source)
            if set(source["artifacts"]) != {"scored_company.parquet"}:
                _reuse_refuse(f"cache lacks its completed company route: {directory}")
            entry = source["artifacts"]["scored_company.parquet"]
            scored_path = directory / "scored_company.parquet"
            if entry.get("path") != scored_path.name or entry.get("sha256") != file_sha256(scored_path):
                _reuse_refuse(f"cache company checksum mismatch: {directory}")
            scored = pd.read_parquet(scored_path)
            if entry.get("rows") != len(scored) or source.get("association_rows") != len(scored):
                _reuse_refuse(f"cache company row mismatch: {directory}")
            original = scored.drop(columns=[*SCORE_COLUMNS, "checkpoint"])
            articles, rejected = _validate_articles(original)
            if len(articles) != len(scored) or rejected["excluded_associations"]:
                _reuse_refuse(f"cache includes timing-rejected associations: {directory}")
            unique = articles[["text_hash", "text"]].drop_duplicates("text_hash").sort_values("text_hash", kind="stable").reset_index(drop=True)
            if source.get("unique_texts") != len(unique):
                _reuse_refuse(f"cache unique-text inventory mismatch: {directory}")
            shards = _load_shards(directory, source, unique, source_shard_size, checkpoint)
            if len(shards) != (len(unique) + source_shard_size - 1) // source_shard_size:
                _reuse_refuse(f"cache shard registry is incomplete: {directory}")
            pd.testing.assert_frame_equal(scored, _route(articles, shards, checkpoint), check_dtype=False)
            if file_sha256(manifest_path) != manifest_sha256 or file_sha256(scored_path) != entry["sha256"]:
                _reuse_refuse(f"cache changed during verification: {directory}")
            for registered in source["shards"]:
                if file_sha256(directory / registered["path"]) != registered["sha256"]:
                    _reuse_refuse(f"cache shard changed during verification: {directory}")
            if shards:
                frames.append(pd.concat(shards, ignore_index=True))
            provenance.append({
                "directory": str(directory), "manifest_sha256": manifest_sha256,
                "signature": source_signature, "unique_texts": len(unique),
                "association_rows": len(articles), "shards": source["shards"],
                "artifacts": source["artifacts"],
            })
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            _reuse_refuse(f"invalid cache {directory} ({error})")
    columns = ["text_hash", *PREDICTION_COLUMNS, "checkpoint"]
    if not frames:
        return pd.DataFrame(columns=columns), provenance
    scores = pd.concat(frames, ignore_index=True)
    distinct = scores.drop_duplicates(columns)
    conflicts = distinct["text_hash"].duplicated(keep=False)
    if conflicts.any():
        text_hash = distinct.loc[conflicts, "text_hash"].iloc[0]
        _reuse_refuse(f"conflicting cached text or scores for text_hash {text_hash}")
    return scores.drop_duplicates("text_hash").set_index("text_hash", drop=False), provenance


def score_fnspid(
    articles_path: str | Path,
    destination: str | Path,
    *,
    checkpoint: FinBERTCheckpoint,
    model_dir: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 16,
    max_length: int = 512,
    shard_size: int = 512,
    resume: bool = False,
    analyzer_factory: Callable[[], Any] | None = None,
    reuse_from: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Score each text once and retain every eligible news/ticker association.

    Resume verifies all committed shards before loading the model and accepts
    only the original input, checkpoint, runtime and inference settings. An
    injected factory with no model directory is an offline test/adapter seam.
    Completed reuse sources are fully verified and fingerprinted into the new
    manifest. Cached texts avoid inference; only missing texts load an analyzer.
    """
    batch_size = _positive_integer(batch_size, "batch_size")
    max_length = _positive_integer(max_length, "max_length")
    shard_size = _positive_integer(shard_size, "shard_size")
    articles_path, destination = Path(articles_path), Path(destination)
    if destination.exists() and not destination.is_dir():
        raise ValueError("Scoring destination must be a directory.")
    populated = destination.exists() and any(path.name != ".collection.lock" for path in destination.iterdir())
    if populated and not resume:
        raise FileExistsError("Scoring destination already exists; use resume or a new version directory.")
    if isinstance(reuse_from, (str, bytes, Path)):
        raise ValueError("reuse_from must be a sequence of scoring directories.")
    if model_dir is None and analyzer_factory is None and not populated and not reuse_from:
        raise ValueError("Scoring requires model_dir or an explicit analyzer_factory.")
    raw_articles = pd.read_parquet(articles_path)
    articles, rejected = _validate_articles(raw_articles)
    unique = articles[["text_hash", "text"]].drop_duplicates("text_hash").sort_values("text_hash", kind="stable").reset_index(drop=True)
    settings = {
        "device": device, "resolved_device": _resolved_device(device, injected=model_dir is None),
        "batch_size": batch_size, "max_length": max_length, "shard_size": shard_size,
    }
    metadata = _model_metadata(model_dir, checkpoint)
    signature = {
        "articles_sha256": file_sha256(articles_path), "checkpoint": checkpoint.identifier,
        "inference_settings": settings, "model_metadata": metadata,
        "library_revision": LIBRARY_REVISION, "runtime_versions": _versions(),
        "timing_policy": "exclude_publication_or_version_timing_uncertain",
    }
    reused_scores, reuse_sources = _load_reuse_sources(reuse_from, signature, checkpoint, destination)
    # Keep the historical signature intact when reuse is not requested.
    if reuse_sources:
        signature["reuse_sources"] = reuse_sources
    if populated and resume:
        _load_manifest(destination / "scoring.manifest.json", signature)
    destination.mkdir(parents=True, exist_ok=True)
    with _output_lock(destination):
        return _score_locked(
            raw_articles, articles, unique, rejected, destination,
            signature=signature, checkpoint=checkpoint, model_dir=model_dir,
            analyzer_factory=analyzer_factory, batch_size=batch_size,
            max_length=max_length, shard_size=shard_size, resume=resume,
            reused_scores=reused_scores, reuse_sources=reuse_sources,
        )


def _score_locked(
    raw_articles: pd.DataFrame, articles: pd.DataFrame, unique: pd.DataFrame,
    rejected: dict[str, int], destination: Path, *, signature: dict[str, Any],
    checkpoint: FinBERTCheckpoint, model_dir: str | Path | None,
    analyzer_factory: Callable[[], Any] | None, batch_size: int,
    max_length: int, shard_size: int, resume: bool,
    reused_scores: pd.DataFrame, reuse_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    populated = any(path.name != ".collection.lock" for path in destination.iterdir())
    if populated and not resume:
        raise FileExistsError("Scoring destination already exists; use resume or a new version directory.")
    settings, metadata = signature["inference_settings"], signature["model_metadata"]
    reused_count = int(unique["text_hash"].isin(reused_scores["text_hash"]).sum())
    manifest_path, scored_path = destination / "scoring.manifest.json", destination / "scored_company.parquet"
    if populated:
        manifest = _load_manifest(manifest_path, signature)
        if manifest.get("unique_texts") != len(unique) or manifest.get("association_rows") != len(articles) or manifest.get("timing_rejections") != rejected:
            _refuse("changed article inventory")
        if reuse_sources and (
            manifest.get("reuse_sources") != reuse_sources
            or manifest.get("reused_unique_texts") != reused_count
            or manifest.get("scored_new_unique_texts") != len(unique) - reused_count
        ):
            _refuse("changed cache reuse inventory")
        _registered_files(destination, manifest)
        shards = _load_shards(destination, manifest, unique, shard_size, checkpoint)
        expected_shards = (len(unique) + shard_size - 1) // shard_size
        if manifest["artifacts"] and len(shards) != expected_shards:
            _refuse("final artifact exists with an incomplete shard registry")
        if manifest["state"] == "complete" and not manifest["artifacts"]:
            _refuse("completed manifest lacks scored-company artifact")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "1.0", "state": "running", "signature": signature,
            "checkpoint": checkpoint.identifier,
            "model": {"repository": checkpoint.repository, "revision": checkpoint.revision, "weights_sha256": checkpoint.weights_sha256, **metadata},
            "input_identifiers": {"articles_sha256": signature["articles_sha256"]},
            "inference_settings": settings, "runtime_versions": signature["runtime_versions"],
            "library_revision": LIBRARY_REVISION, "input_association_rows": len(raw_articles),
            "unique_texts": len(unique), "association_rows": len(articles),
            "timing_rejections": rejected, "shards": [], "artifacts": {},
            "coverage_note": "FNSPID publication dates and associations establish no historical availability or exhaustive coverage.",
        }
        if reuse_sources:
            manifest.update({
                "reuse_sources": reuse_sources, "reused_unique_texts": reused_count,
                "scored_new_unique_texts": len(unique) - reused_count,
            })
        _write_json(manifest_path, manifest)
        shards = []
    shard_count = (len(unique) + shard_size - 1) // shard_size
    if len(shards) < shard_count:
        analyzer = None
        for index in range(len(shards), shard_count):
            start, end = index * shard_size, min((index + 1) * shard_size, len(unique))
            inventory = unique.iloc[start:end]
            missing = inventory.loc[~inventory["text_hash"].isin(reused_scores["text_hash"])]
            pieces = []
            cached = inventory.loc[inventory["text_hash"].isin(reused_scores["text_hash"])]
            if not cached.empty:
                cache_frame = reused_scores.loc[cached["text_hash"].tolist()].reset_index(drop=True)
                if cache_frame["text"].tolist() != cached["text"].tolist():
                    _reuse_refuse("target text differs from cached text with the same hash")
                pieces.append(cache_frame)
            if not missing.empty:
                if analyzer is None:
                    if analyzer_factory is not None:
                        analyzer = analyzer_factory()
                    elif model_dir is not None:
                        analyzer = load_local_finbert(
                            model_dir, checkpoint=checkpoint, device=settings["resolved_device"],
                            batch_size=batch_size, max_length=max_length,
                        )
                    else:
                        raise ValueError("Remaining scoring shards require model_dir or analyzer_factory.")
                texts = missing["text"].tolist()
                batches = [_predict(analyzer, texts[offset:offset + batch_size]) for offset in range(0, len(texts), batch_size)]
                inferred = pd.concat(batches, ignore_index=True)
                inferred.insert(0, "text_hash", missing["text_hash"].to_numpy())
                inferred["checkpoint"] = checkpoint.identifier
                pieces.append(inferred)
            shard_frame = pd.concat(pieces, ignore_index=True).set_index("text_hash", drop=False).loc[inventory["text_hash"].tolist()].reset_index(drop=True)
            _commit_frame(shard_frame, destination, manifest_path, manifest, {
                "path": f"shards/part-{index:06d}.parquet", "start": start, "end": end,
            }, shard=True)
            shards.append(shard_frame)
    routed = _route(articles, shards, checkpoint)
    if "scored_company.parquet" in manifest["artifacts"]:
        entry = manifest["artifacts"]["scored_company.parquet"]
        if entry.get("path") != scored_path.name or entry.get("sha256") != file_sha256(scored_path) or entry.get("rows") != len(routed):
            _refuse("scored-company checksum or row mismatch")
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(scored_path), routed, check_dtype=False)
        except (AssertionError, OSError, ValueError) as error:
            _refuse(f"changed scored-company content ({error})")
    else:
        if manifest["state"] == "complete":
            _refuse("completed manifest lacks scored-company artifact")
        _commit_frame(routed, destination, manifest_path, manifest, {"path": scored_path.name}, shard=False)
    if manifest["state"] != "complete":
        manifest["state"] = "complete"
        _write_json(manifest_path, manifest)
    return {"scored_path": scored_path, "manifest_path": manifest_path, "manifest": manifest}


__all__ = ["score_fnspid"]
