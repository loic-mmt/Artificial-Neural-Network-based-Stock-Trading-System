"""Integrity, portability and effective training identity for multimodal tasks."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import numpy as np
import pytest

from trading_system.artifacts import multimodal_study as artifacts


@dataclass
class _Settings:
    seed: int = 7
    width: int = 32


def _spec():
    return {
        "data": {"dataset_path": "/original/prices.parquet", "sha256": "a" * 64},
        "cv": {"initial_train_fraction": .5, "inner_val_fraction": .2,
               "train_start": "2005-01-03", "outer_start": "2018-01-02"},
        "tickers": ["A", "B"],
        "preprocessing": {"columns": ["f1", "f2"], "fills": [0., 1.]},
        "model": _Settings(),
        "runtime": {"resolved_device": "cpu", "deterministic": True,
                    "source_sha256": "b" * 64},
        "ablation": {"gnn_layers": 1, "candidates": ["gru", "identity"]},
        "output_dir": "/original/output", "study_name": "first",
    }


def test_strict_canonical_json_normalizes_scientific_values_preserving_order():
    value = {"b": np.array([1., 2.]), "a": _Settings(),
             "date": datetime(2020, 1, 1, tzinfo=timezone.utc)}
    assert json.loads(artifacts.canonical_json(value))["a"] == {"seed": 7, "width": 32}
    assert artifacts.stable_digest(value) == artifacts.stable_digest(dict(reversed(list(value.items()))))
    assert artifacts.stable_digest(["A", "B"]) != artifacts.stable_digest(["B", "A"])
    for invalid in (float("nan"), np.array([float("inf")]), {1: "ambiguous"}, {"x": object()}):
        with pytest.raises((TypeError, ValueError)):
            artifacts.canonical_json(invalid)


def test_signature_excludes_only_bookkeeping_and_location():
    first = _spec()
    relocated = deepcopy(first)
    relocated["data"]["dataset_path"] = "C:/relocated/prices.parquet"
    relocated["output_dir"] = "C:/other/output"
    relocated["study_name"] = "second"
    relocated["ablation"]["candidates"] = ["sector"]
    relocated["code_commit"] = "different bookkeeping"
    relocated["working_tree_dirty"] = True
    relocated["runtime"]["observed_torch_state"] = {"deterministic_algorithms": False}
    assert artifacts.training_signature(first) == artifacts.training_signature(relocated)


@pytest.mark.parametrize("field", ["cv", "code", "data", "tickers", "model", "device", "preprocessing"])
def test_effective_changes_invalidate_training_signature(field):
    spec = _spec()
    changed = deepcopy(spec)
    if field == "cv":
        changed["cv"]["initial_train_fraction"] = .6
    elif field == "code":
        changed["runtime"]["source_sha256"] = "c" * 64
    elif field == "data":
        changed["data"]["sha256"] = "d" * 64
    elif field == "tickers":
        changed["tickers"] = ["B", "A"]
    elif field == "model":
        changed["ablation"]["gnn_layers"] = 2
    elif field == "device":
        changed["runtime"]["resolved_device"] = "mps"
    else:
        changed["preprocessing"]["fills"] = [1., 0.]
    assert artifacts.training_signature(spec) != artifacts.training_signature(changed)


def test_json_atomic_failure_preserves_previous_artifact(tmp_path, monkeypatch):
    target = tmp_path / "metadata.json"
    artifacts.atomic_write_json(target, {"version": 1})
    with pytest.raises(ValueError):
        artifacts.atomic_write_json(target, {"bad": float("nan")})
    assert json.loads(target.read_text()) == {"version": 1}

    def fail_replace(*args):
        raise OSError("simulated interrupted rename")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        artifacts.atomic_write_json(target, {"version": 2})
    assert json.loads(target.read_text()) == {"version": 1}
    assert set(tmp_path.iterdir()) == {target}


def test_torch_atomic_writer_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    target = tmp_path / "model.pt"
    artifacts.atomic_torch_save(target, {"weights": torch.tensor([1., 2.])})
    previous_hash = artifacts.sha256_file(target)

    def fail_save(value, stream):
        stream.write(b"partial checkpoint")
        raise OSError("simulated interrupted checkpoint")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interrupted"):
        artifacts.atomic_torch_save(target, {})
    assert artifacts.sha256_file(target) == previous_hash
    assert set(tmp_path.iterdir()) == {target}


def test_text_atomic_write_preserves_utf8_and_old_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "report.md"
    content = "# Résultats\n\nΔ Sharpe : 0,05\n"
    artifacts.atomic_write_text(target, content)
    assert target.read_bytes() == content.encode("utf-8")
    with pytest.raises(TypeError, match="string"):
        artifacts.atomic_write_text(target, b"invalid bytes")

    def fail_replace(*args):
        raise OSError("simulated interrupted text write")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        artifacts.atomic_write_text(target, "replacement")
    assert target.read_bytes() == content.encode("utf-8")
    assert set(tmp_path.iterdir()) == {target}


def test_completion_reuse_is_portable_and_detects_corruption(tmp_path):
    source, relocated = tmp_path / "source", tmp_path / "relocated"
    source.mkdir()
    relocated.mkdir()
    artifacts.atomic_write_json(source / "model.json", {"weights": [1., 2.]})
    record = artifacts.file_record("model.json", source)
    assert record["path"] == "model.json"
    signature = artifacts.training_signature(_spec())
    manifest = artifacts.completion_manifest(signature, [record])
    (relocated / "model.json").write_bytes((source / "model.json").read_bytes())
    assert artifacts.validate_completion(manifest, relocated, signature) == (relocated / "model.json",)
    assert artifacts.verify_reuse(manifest, relocated, signature)["verified"] is True
    with pytest.raises(ValueError, match="signature"):
        artifacts.verify_reuse(manifest, relocated, "f" * 64)
    (relocated / "model.json").write_bytes(b"x" * record["size_bytes"])
    with pytest.raises(ValueError, match="integrity"):
        artifacts.validate_completion(manifest, relocated, signature)
    (relocated / "model.json").unlink()
    with pytest.raises(ValueError, match="Missing"):
        artifacts.validate_completion(manifest, relocated, signature)


@pytest.mark.parametrize("reference", ["", "/absolute", "../escape", "x/../escape", "x//file",
                                        "./file", "C:/absolute", "C:drive-relative", "x\\file",
                                        "NUL", "weights/NUL.pt", "weights/a:b", "model?.pt", "file."])
def test_artifact_reference_rejects_unportable_or_escaping_paths(tmp_path, reference):
    with pytest.raises(ValueError):
        artifacts.artifact_path(reference, tmp_path)


def test_file_validation_rejects_duplicate_records_and_symlink_escape(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    artifacts.atomic_write_json(root / "weights.json", {"x": 1})
    record = artifacts.file_record("weights.json", root)
    with pytest.raises(ValueError, match="Duplicate"):
        artifacts.validate_files([record, record], root)
    outside = tmp_path / "outside.json"
    artifacts.atomic_write_json(outside, {"x": 2})
    link = root / "escape"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")
    with pytest.raises(ValueError, match="escapes"):
        artifacts.artifact_path("escape", root)
    with pytest.raises(ValueError, match="within"):
        artifacts.file_record(outside, root)


def test_status_without_complete_files_cannot_be_reused(tmp_path):
    signature = artifacts.training_signature(_spec())
    with pytest.raises(ValueError, match="files"):
        artifacts.completion_manifest(signature, [])
    with pytest.raises(ValueError, match="missing or invalid"):
        artifacts.validate_completion({"schema_version": 1, "status": "running",
                                       "training_signature": signature, "files": []}, tmp_path)
    with pytest.raises(ValueError, match="files"):
        artifacts.validate_completion({"schema_version": 1, "status": "complete",
                                       "training_signature": signature, "files": []}, tmp_path)


def test_source_provenance_survives_relocation_and_detects_source_changes(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        source = root / "src" / "trading_system"
        source.mkdir(parents=True)
        (source / "model.py").write_text("width = 32\n")
    first, second = [artifacts.runtime_provenance(root) for root in roots]
    assert first["source_sha256"] == second["source_sha256"]
    assert first["sources"] == [{"path": "src/trading_system/model.py",
                                 "sha256": artifacts.sha256_file(roots[0] / "src/trading_system/model.py")}]
    (roots[1] / "src/trading_system/model.py").write_text("width = 64\n")
    assert artifacts.runtime_provenance(roots[1])["source_sha256"] != first["source_sha256"]


def test_transient_global_torch_flags_do_not_change_effective_signature(tmp_path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "src" / "trading_system"
    source.mkdir(parents=True)
    (source / "model.py").write_text("width = 32\n")
    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        first = artifacts.runtime_provenance(tmp_path)
        torch.use_deterministic_algorithms(True)
        second = artifacts.runtime_provenance(tmp_path)
        assert first["runtime"]["observed_torch_state"] != second["runtime"]["observed_torch_state"]
        assert artifacts.training_signature({"provenance": first, "deterministic": True}) == artifacts.training_signature({"provenance": second, "deterministic": True})
    finally:
        torch.use_deterministic_algorithms(old)


def test_preprocessing_state_captures_frozen_arrays_and_selector_state():
    prepared = SimpleNamespace(
        columns=("f1", "f2"), tickers=("A", "B"), fills={"f1": 0., "f2": 1.},
        scaler=SimpleNamespace(mean_=np.array([0., 1.]), scale_=np.array([1., 2.])),
        selector=SimpleNamespace(state_dict=lambda: {"columns": ["f1", "f2"]}),
        overfitting_selector=None, fracdiff=None, purging={"kept": 10},
    )
    saved = artifacts.preprocessing_state(prepared)
    assert saved["scaler"] == {"mean": [0., 1.], "scale": [1., 2.]}
    assert saved["tickers"] == ["A", "B"]
    assert saved["feature_selector"]["columns"] == ["f1", "f2"]
    assert saved["market_scaler"] is None
    prepared.scaler.mean_[0] = 99
    assert saved["scaler"]["mean"][0] == 0
