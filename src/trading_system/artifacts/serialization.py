from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

_CLASS_NAMES = ("Sell", "Hold", "Buy")
_ARRAY_MARKER = "__artifact_array__"
_TUPLE_MARKER = "__artifact_tuple__"


def to_jsonable(value: Any) -> Any:
    """Return deterministic, JSON-safe metadata without unstable fallbacks."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("JSON metadata cannot contain NaN or infinity.")
        return value
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_jsonable(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        raise TypeError("NumPy arrays belong in binary artifact state, not JSON.")
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key in sorted(value, key=lambda candidate: str(candidate)):
            if not isinstance(key, str):
                raise TypeError("JSON metadata mapping keys must be strings.")
            converted[key] = to_jsonable(value[key])
        return converted
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"Unsupported JSON metadata type: {type(value).__name__}.")


def stable_config_hash(value: Any) -> str:
    """Hash canonical UTF-8 JSON with SHA-256."""

    encoded = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_config(
    model_name: str,
    model_parameters: Mapping[str, Any],
    experiment_parameters: Mapping[str, Any],
    decision_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_parameters": model_parameters,
        "experiment_parameters": experiment_parameters,
        "decision_parameters": decision_parameters,
    }


@dataclass(frozen=True)
class ArtifactManifest:
    """Portable metadata required to reconstruct a trained experiment."""

    format_version: int
    model_name: str
    model_parameters: dict[str, Any]
    experiment_parameters: dict[str, Any]
    config_hash: str
    feature_columns: tuple[str, ...]
    context_len: int
    class_names: tuple[str, ...] = _CLASS_NAMES
    decision_parameters: dict[str, Any] = field(default_factory=dict)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version <= 0
        ):
            raise ValueError("format_version must be a positive integer.")
        if (
            isinstance(self.context_len, bool)
            or not isinstance(self.context_len, int)
            or self.context_len <= 0
        ):
            raise ValueError("context_len must be a positive integer.")
        name = str(self.model_name).strip().lower()
        if not name:
            raise ValueError("model_name must be non-empty.")
        features = tuple(str(column).strip() for column in self.feature_columns)
        if not features or any(not column for column in features):
            raise ValueError("feature_columns must contain non-empty names.")
        if len(features) != len(set(features)):
            raise ValueError("feature_columns must be unique and ordered.")
        classes = tuple(self.class_names)
        if classes != _CLASS_NAMES:
            raise ValueError(f"class_names must equal {_CLASS_NAMES}.")

        model_parameters = to_jsonable(self.model_parameters)
        experiment_parameters = to_jsonable(self.experiment_parameters)
        decision_parameters = to_jsonable(self.decision_parameters)
        runtime_metadata = to_jsonable(self.runtime_metadata)
        for field_name, value in (
            ("model_parameters", model_parameters),
            ("experiment_parameters", experiment_parameters),
            ("decision_parameters", decision_parameters),
            ("runtime_metadata", runtime_metadata),
        ):
            if not isinstance(value, dict):
                raise TypeError(f"{field_name} must be a mapping.")

        expected_hash = stable_config_hash(
            _manifest_config(
                name,
                model_parameters,
                experiment_parameters,
                decision_parameters,
            )
        )
        if self.config_hash != expected_hash:
            raise ValueError("config_hash does not match canonical manifest config.")

        object.__setattr__(self, "model_name", name)
        object.__setattr__(self, "feature_columns", features)
        object.__setattr__(self, "class_names", classes)
        object.__setattr__(self, "model_parameters", model_parameters)
        object.__setattr__(self, "experiment_parameters", experiment_parameters)
        object.__setattr__(self, "decision_parameters", decision_parameters)
        object.__setattr__(self, "runtime_metadata", runtime_metadata)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _is_torch_tensor(value: Any) -> bool:
    module = type(value).__module__.split(".", 1)[0]
    return module == "torch" and all(
        hasattr(value, attribute) for attribute in ("detach", "cpu", "numpy")
    )


def _encode_binary_state(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    tensor = _is_torch_tensor(value)
    if tensor:
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be stored in model artifacts.")
        key = f"array_{len(arrays):06d}"
        arrays[key] = np.ascontiguousarray(value)
        return {_ARRAY_MARKER: key, "kind": "torch" if tensor else "numpy"}
    if isinstance(value, Mapping):
        encoded: dict[str, Any] = {}
        for key in sorted(value, key=lambda candidate: str(candidate)):
            if not isinstance(key, str):
                raise TypeError("Artifact state mapping keys must be strings.")
            encoded[key] = _encode_binary_state(value[key], arrays)
        return encoded
    if isinstance(value, tuple):
        return {_TUPLE_MARKER: [_encode_binary_state(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_encode_binary_state(item, arrays) for item in value]
    return to_jsonable(value)


def _write_binary_state(directory: Path, name: str, state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"{name} state must be a mapping.")
    arrays: dict[str, np.ndarray] = {}
    metadata = _encode_binary_state(state, arrays)
    _write_bytes(directory / f"{name}.json", _json_bytes(metadata))
    path = directory / f"{name}.npz"
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_model_artifact(
    destination: str | Path,
    *,
    manifest: ArtifactManifest,
    model_state: Mapping[str, Any],
    scaler_state: Mapping[str, Any],
    training_history: Mapping[str, Any],
    metrics: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    """Atomically save validated metadata and non-pickle numerical state."""

    if not isinstance(manifest, ArtifactManifest):
        raise TypeError("manifest must be an ArtifactManifest.")
    destination_path = Path(destination).expanduser().resolve()
    if destination_path == destination_path.parent:
        raise ValueError("Artifact destination cannot be a filesystem root.")
    parent = destination_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"Artifact already exists: {destination_path}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.tmp-", dir=parent)
    )
    backup: Path | None = None
    try:
        _write_bytes(
            temporary / "manifest.json", _json_bytes(dataclasses.asdict(manifest))
        )
        _write_bytes(temporary / "training_history.json", _json_bytes(training_history))
        _write_bytes(temporary / "metrics.json", _json_bytes(metrics))
        _write_binary_state(temporary, "model_state", model_state)
        _write_binary_state(temporary, "scaler_state", scaler_state)
        files = sorted(path.name for path in temporary.iterdir() if path.is_file())
        inventory = {
            "format_version": manifest.format_version,
            "files": {name: _sha256_file(temporary / name) for name in files},
        }
        _write_bytes(temporary / "inventory.json", _json_bytes(inventory))
        _fsync_directory(temporary)

        if destination_path.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination_path.name}.backup-", dir=parent
                )
            )
            backup.rmdir()
            os.replace(destination_path, backup)
        try:
            os.replace(temporary, destination_path)
        except Exception:
            if (
                backup is not None
                and backup.exists()
                and not destination_path.exists()
            ):
                os.replace(backup, destination_path)
            raise
        _fsync_directory(parent)
        if backup is not None:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        return destination_path
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid artifact JSON: {path.name}") from error


def _decode_binary_state(
    value: Any, arrays: Mapping[str, np.ndarray], device: str
) -> Any:
    if isinstance(value, dict) and set(value) == {_ARRAY_MARKER, "kind"}:
        key = value[_ARRAY_MARKER]
        if key not in arrays:
            raise ValueError(f"Artifact state references missing array: {key}")
        array = np.array(arrays[key], copy=True)
        if value["kind"] == "numpy":
            return array
        if value["kind"] != "torch":
            raise ValueError("Artifact state contains unknown array kind.")
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError(
                'Loading PyTorch state requires `pip install -e ".[neural]"`.'
            ) from error
        return torch.from_numpy(array).to(device)
    if isinstance(value, dict) and set(value) == {_TUPLE_MARKER}:
        return tuple(
            _decode_binary_state(item, arrays, device)
            for item in value[_TUPLE_MARKER]
        )
    if isinstance(value, dict):
        return {
            key: _decode_binary_state(item, arrays, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_binary_state(item, arrays, device) for item in value]
    return value


def _load_binary_state(source: Path, name: str, device: str) -> dict[str, Any]:
    metadata = _read_json(source / f"{name}.json")
    try:
        with np.load(source / f"{name}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid artifact binary state: {name}") from error
    decoded = _decode_binary_state(metadata, arrays, device)
    if not isinstance(decoded, dict):
        raise ValueError(f"Artifact {name} must decode to a mapping.")
    return decoded


def load_model_artifact(
    source: str | Path,
    *,
    expected_format_versions: Sequence[int] = (1,),
    map_device: str = "cpu",
) -> tuple[ArtifactManifest, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify checksums before safely loading metadata and numerical state."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {source_path}")
    versions = tuple(expected_format_versions)
    if not versions or any(
        isinstance(value, bool) or int(value) <= 0 for value in versions
    ):
        raise ValueError("expected_format_versions must contain positive integers.")
    if map_device not in ("cpu", "cuda", "mps"):
        raise ValueError("map_device must be 'cpu', 'cuda', or 'mps'.")

    inventory = _read_json(source_path / "inventory.json")
    if not isinstance(inventory, dict) or set(inventory) != {
        "format_version",
        "files",
    }:
        raise ValueError("Invalid artifact inventory.")
    files = inventory["files"]
    if not isinstance(files, dict):
        raise ValueError("Invalid artifact file inventory.")
    expected_names = {
        "manifest.json",
        "training_history.json",
        "metrics.json",
        "model_state.json",
        "model_state.npz",
        "scaler_state.json",
        "scaler_state.npz",
    }
    if set(files) != expected_names:
        raise ValueError("Artifact file inventory is incomplete or unexpected.")
    actual_names = {path.name for path in source_path.iterdir() if path.is_file()}
    if actual_names != expected_names | {"inventory.json"}:
        raise ValueError("Artifact directory contains unlisted or missing files.")
    for name, expected_digest in files.items():
        if (
            not isinstance(expected_digest, str)
            or _sha256_file(source_path / name) != expected_digest
        ):
            raise ValueError(f"Artifact checksum mismatch: {name}")

    manifest_payload = _read_json(source_path / "manifest.json")
    try:
        manifest = ArtifactManifest(**manifest_payload)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid artifact manifest.") from error
    supported = set()
    try:
        supported = {int(value) for value in versions}
    except (TypeError, ValueError) as error:
        raise ValueError(
            "expected_format_versions must contain positive integers."
        ) from error
    if manifest.format_version not in supported:
        raise ValueError(
            f"Unsupported artifact format_version: {manifest.format_version}"
        )
    if inventory["format_version"] != manifest.format_version:
        raise ValueError("Inventory and manifest format versions differ.")

    model_state = _load_binary_state(source_path, "model_state", map_device)
    scaler_state = _load_binary_state(source_path, "scaler_state", "cpu")
    diagnostics = {
        "training_history": _read_json(source_path / "training_history.json"),
        "metrics": _read_json(source_path / "metrics.json"),
    }
    return manifest, model_state, scaler_state, diagnostics


def validate_artifact_compatibility(
    manifest: ArtifactManifest,
    *,
    model_name: str,
    feature_columns: Sequence[str],
    context_len: int,
    class_names: Sequence[str] = _CLASS_NAMES,
) -> None:
    """Reject every reconstruction mismatch before model state is loaded."""

    expected = {
        "model_name": str(model_name).strip().lower(),
        "feature_columns": tuple(feature_columns),
        "context_len": context_len,
        "class_names": tuple(class_names),
    }
    actual = {
        "model_name": manifest.model_name,
        "feature_columns": manifest.feature_columns,
        "context_len": manifest.context_len,
        "class_names": manifest.class_names,
    }
    mismatches = [
        f"{name}: expected {expected[name]!r}, artifact has {actual[name]!r}"
        for name in expected
        if expected[name] != actual[name]
    ]
    if mismatches:
        raise ValueError(
            "Artifact compatibility mismatch; " + "; ".join(mismatches)
        )


__all__ = [
    "ArtifactManifest",
    "load_model_artifact",
    "save_model_artifact",
    "stable_config_hash",
    "to_jsonable",
    "validate_artifact_compatibility",
]
