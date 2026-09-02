"""Safe, reproducible trained-model artifacts."""

from .serialization import (
    ArtifactManifest,
    load_model_artifact,
    save_model_artifact,
    stable_config_hash,
    to_jsonable,
    validate_artifact_compatibility,
)


def __getattr__(name: str):
    if name in {
        "build_experiment_manifest",
        "hash_dataframe",
        "save_experiment_artifact",
    }:
        from . import experiment

        return getattr(experiment, name)
    raise AttributeError(name)

__all__ = [
    "ArtifactManifest",
    "build_experiment_manifest",
    "hash_dataframe",
    "load_model_artifact",
    "save_model_artifact",
    "save_experiment_artifact",
    "stable_config_hash",
    "to_jsonable",
    "validate_artifact_compatibility",
]
