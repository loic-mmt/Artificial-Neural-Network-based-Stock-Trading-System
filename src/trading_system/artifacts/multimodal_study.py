"""Portable, verified artifacts for matched multimodal training tasks.

Signatures describe effective training inputs, not the directory or study that
owns them. Completion and reuse always verify the referenced file bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
_HASH = re.compile(r"^[0-9a-f]{64}$")
_BOOKKEEPING = frozenset({
    "output_dir", "output_path", "destination", "run_dir", "reference_run",
    "study_name", "study_id", "variant_id", "candidates", "graph_candidates",
    "created_at", "updated_at", "code_commit", "working_tree_dirty", "git_dirty",
    "observed_torch_state",
})


def _json_value(value: Any) -> Any:
    """Normalize common scientific metadata, rejecting ambiguous/nonfinite values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON metadata keys must be strings.")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON metadata cannot contain nonfinite numbers.")
        return value
    raise TypeError(f"Unsupported JSON metadata type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Deterministic strict JSON; sequence order remains semantically significant."""
    return json.dumps(_json_value(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: str | Path, writer) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    # Serialize before creating a file: malformed metadata cannot replace an
    # existing valid artifact or leave a partially written JSON document.
    content = (canonical_json(value) + "\n").encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(content))


def atomic_write_text(path: str | Path, value: str) -> None:
    """Atomically persist exact UTF-8 text, preserving intentional newlines."""
    if not isinstance(value, str):
        raise TypeError("Text artifact content must be a string.")
    content = value.encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(content))


def atomic_torch_save(path: str | Path, value: Any) -> None:
    """Persist a checkpoint atomically without requiring torch at module import."""
    import torch

    _atomic_write(path, lambda stream: torch.save(value, stream))


def artifact_path(reference: str, root: str | Path) -> Path:
    """Resolve one portable relative reference without escaping the artifact root."""
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise ValueError("Artifact references must be non-empty relative POSIX paths.")
    if (reference.startswith("/") or PureWindowsPath(reference).drive
            or any(part in ("", ".", "..") for part in reference.split("/"))):
        raise ValueError(f"Unsafe artifact reference: {reference!r}")
    for part in reference.split("/"):
        if (any(character in '<>:"|?*' or ord(character) < 32 for character in part)
                or part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved()):
            raise ValueError(f"Unportable artifact reference: {reference!r}")
    base = Path(root).resolve()
    result = (base / reference).resolve()
    if not result.is_relative_to(base):
        raise ValueError(f"Artifact reference escapes its root: {reference!r}")
    return result


def file_record(path: str | Path, root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    supplied = Path(path)
    actual = (supplied if supplied.is_absolute() else base / supplied).resolve()
    if not actual.is_relative_to(base) or not actual.is_file():
        raise ValueError(f"Artifact must be an existing file within its root: {path}")
    reference = actual.relative_to(base).as_posix()
    artifact_path(reference, base)
    return {"path": reference, "size_bytes": actual.stat().st_size,
            "sha256": sha256_file(actual)}


def validate_files(records: Sequence[Mapping[str, Any]], root: str | Path) -> tuple[Path, ...]:
    """Check uniqueness, containment, sizes and hashes; never trust status alone."""
    if not records:
        raise ValueError("A completed task requires artifact files.")
    paths, seen = [], set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Invalid artifact file record.")
        reference = record.get("path")
        path = artifact_path(reference, root)
        if reference in seen:
            raise ValueError(f"Duplicate artifact reference: {reference}")
        seen.add(reference)
        size, checksum = record.get("size_bytes"), record.get("sha256")
        if (isinstance(size, bool) or not isinstance(size, int) or size < 0
                or not isinstance(checksum, str) or not _HASH.fullmatch(checksum)):
            raise ValueError(f"Invalid size/hash for artifact: {reference}")
        if not path.is_file():
            raise ValueError(f"Missing artifact file: {reference}")
        if path.stat().st_size != size or sha256_file(path) != checksum:
            raise ValueError(f"Artifact integrity mismatch: {reference}")
        paths.append(path)
    return tuple(paths)


def runtime_provenance(root: str | Path | None = None) -> dict[str, Any]:
    """Fingerprint source bytes and numerical runtime, independent of checkout path.

    Every Python source in src/trading_system is included conservatively. Git
    dirty status and commit bookkeeping do not substitute for source hashes.
    Caller records requested/resolved device and training precision in its spec.
    """
    base = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[3]
    source_root = base / "src" / "trading_system"
    sources = [{"path": path.relative_to(base).as_posix(), "sha256": sha256_file(path)}
               for path in sorted(source_root.rglob("*.py"))]
    if not sources:
        raise ValueError(f"No training Python sources found under {source_root}")
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "torch", "scipy", "scikit-learn",
                 "TA-Lib", "statsmodels"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    processor = platform.processor()
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True, timeout=3, check=False)
            if result.returncode == 0 and result.stdout.strip():
                processor = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    runtime = {"python": platform.python_version(), "implementation": sys.implementation.name,
               "system": platform.system(), "release": platform.release(),
               "machine": platform.machine(), "processor": processor,
               "cpu_count": os.cpu_count(),
               "packages": packages}
    if packages["torch"] is not None:
        import torch

        mps = getattr(torch.backends, "mps", None)
        runtime["torch"] = {
            "default_dtype": str(torch.get_default_dtype()),
            "num_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "mps_available": bool(mps is not None and mps.is_available()),
            "cuda_devices": [torch.cuda.get_device_name(index)
                             for index in range(torch.cuda.device_count())],
        }
        runtime["observed_torch_state"] = {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
    return {"schema_version": SCHEMA_VERSION, "sources": sources,
            "source_sha256": stable_digest(sources), "runtime": runtime}


def _effective_spec(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _effective_spec(item) for key, item in value.items()
                if key not in _BOOKKEEPING and not key.endswith("_path")}
    if isinstance(value, list):
        return [_effective_spec(item) for item in value]
    return value


def training_signature(spec: Mapping[str, Any]) -> str:
    """Hash an effective training spec after removing explicit path/bookkeeping keys.

    Include data/context SHA256, exact CV dates, ticker order, preprocessing,
    effective model settings, seed and runtime_provenance. File locations use
    named *_path keys; content hashes must accompany them. Source-file relative
    'path' fields are retained, as source layout can change import behavior.
    """
    if not isinstance(spec, Mapping) or not spec:
        raise ValueError("Training signature requires a non-empty mapping.")
    return stable_digest({"schema_version": SCHEMA_VERSION,
                          "training": _effective_spec(_json_value(spec))})


def preprocessing_state(prepared, market_scaler=None, market_columns=()) -> dict[str, Any]:
    """Persist the frozen train-fitted state without importing experiment runners."""
    def scaler_state(scaler):
        if scaler is None:
            return None
        if scaler.mean_ is None or scaler.scale_ is None:
            raise ValueError("Cannot save an unfitted preprocessing scaler.")
        return {"mean": scaler.mean_, "scale": scaler.scale_}

    def fitted_state(value):
        return value.state_dict() if value is not None else None

    fills = prepared.fills.to_dict() if hasattr(prepared.fills, "to_dict") else dict(prepared.fills)
    return _json_value({
        "schema_version": SCHEMA_VERSION,
        "feature_columns": prepared.columns, "tickers": prepared.tickers,
        "fill_values": fills, "scaler": scaler_state(prepared.scaler),
        "feature_selector": fitted_state(prepared.selector),
        "overfitting_selector": fitted_state(prepared.overfitting_selector),
        "fracdiff": fitted_state(prepared.fracdiff),
        "purging": prepared.purging,
        "market_columns": market_columns, "market_scaler": scaler_state(market_scaler),
    })


def completion_manifest(signature: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(signature, str) or not _HASH.fullmatch(signature):
        raise ValueError("Completion requires a SHA256 training signature.")
    if not records:
        raise ValueError("Completion requires artifact files.")
    return _json_value({"schema_version": SCHEMA_VERSION, "status": "complete",
                        "training_signature": signature, "files": list(records)})


def validate_completion(manifest: Mapping[str, Any], root: str | Path,
                        expected_signature: str | None = None) -> tuple[Path, ...]:
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported task completion manifest.")
    signature = manifest.get("training_signature")
    if (manifest.get("status") != "complete" or not isinstance(signature, str)
            or not _HASH.fullmatch(signature)):
        raise ValueError("Task completion is missing or invalid.")
    if expected_signature is not None and signature != expected_signature:
        raise ValueError("Reuse training signature does not match the requested task.")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("Task completion requires a file manifest.")
    return validate_files(records, root)


def verify_reuse(manifest: Mapping[str, Any], root: str | Path,
                 expected_signature: str) -> dict[str, Any]:
    validate_completion(manifest, root, expected_signature)
    return {"training_signature": expected_signature, "source_root": str(Path(root).resolve()),
            "source_manifest_sha256": stable_digest(manifest),
            "files": _json_value(manifest["files"]), "verified": True}


__all__ = [
    "SCHEMA_VERSION", "canonical_json", "stable_digest", "sha256_file",
    "atomic_write_json", "atomic_write_text", "atomic_torch_save", "artifact_path", "file_record",
    "validate_files", "runtime_provenance", "training_signature", "preprocessing_state",
    "completion_manifest", "validate_completion", "verify_reuse",
]
