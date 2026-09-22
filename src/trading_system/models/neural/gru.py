from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trading_system.models.specs import ModelBuildContext

from .gru_base import (
    GRUVariantClassifier,
    build_gru_module,
    create_gru_variant_classifier,
)


class GRUClassifier(GRUVariantClassifier):
    model_name = "gru"


def create_gru_classifier(
    context: ModelBuildContext,
    parameters: Mapping[str, Any],
) -> GRUClassifier:
    return create_gru_variant_classifier(context, parameters, GRUClassifier)


__all__ = ["GRUClassifier", "build_gru_module", "create_gru_classifier"]
