"""Binary short detector, shared dual heads, or independent GRU branches.

These models belong to the isolated short-decision study, not the registry of
historical three-class runners. Financial activations are not calibrated odds.
"""

from dataclasses import dataclass

from .config import GRUConfig
from .gru_base import GRUVariantClassifier, build_gru_module, create_gru_variant_classifier

SHORT_ARCHITECTURES = ("short_detector", "dual_head", "separate_grus")


@dataclass(frozen=True)
class ShortDecisionGRUConfig(GRUConfig):
    architecture: str = "short_detector"

    def __post_init__(self):
        super().__post_init__()
        if self.architecture not in SHORT_ARCHITECTURES or self.head_type != "linear":
            raise ValueError("Short study requires a known architecture and linear heads.")


class ShortDecisionGRUClassifier(GRUVariantClassifier):
    model_name = "gru_short_decision"
    state_size = 10

    @property
    def head_names(self):
        return ("short",) if self.config.architecture == "short_detector" else ("long", "short")

    def _build_module(self, torch, nn):
        context, config = self.context, self.gru_config
        heads = 1 if config.architecture == "short_detector" else 2
        encoders = 2 if config.architecture == "separate_grus" else 1

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoders = nn.ModuleList([build_gru_module(context, config, torch, nn,
                    head_factory=lambda width, classes: nn.Identity()) for _ in range(encoders)])
                self.heads = nn.ModuleList([nn.Linear(self.encoders[0].embedding_size + 10, 1)
                                           for _ in range(heads)])

            def encode(self, X):
                return torch.stack([encoder.encode(X) for encoder in self.encoders], dim=1)

            def head_logits(self, embeddings, states):
                if states.shape != (len(embeddings), 10):
                    raise ValueError("Current states require shape [batch, 10].")
                return torch.cat([head(torch.cat((embeddings[:, i if encoders == 2 else 0],
                                                  states.detach()), dim=1))
                                  for i, head in enumerate(self.heads)], dim=1)

            def forward(self, X, states):
                return self.head_logits(self.encode(X), states)

        return Module()

    def fit(self, *args, **kwargs):
        raise RuntimeError("Use fit_short_decision_model with chronological panels.")

    def predict_proba(self, *args, **kwargs):
        raise RuntimeError("Use rollout_short_decisions with a fresh trading book.")


def create_short_decision_gru(context, parameters, architecture):
    return create_gru_variant_classifier(context, {**parameters, "architecture": architecture},
        ShortDecisionGRUClassifier, config_type=ShortDecisionGRUConfig)
