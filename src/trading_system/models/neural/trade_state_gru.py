"""Isolated GRU encoder plus current detached trading-state head."""

from .gru_base import GRUVariantClassifier, build_gru_module, create_gru_variant_classifier


class TradeStateGRUClassifier(GRUVariantClassifier):
    model_name = "gru_trade_state"
    state_size = 10

    def _build_module(self, torch_module, nn_module):
        context, config, state_size = self.context, self.gru_config, self.state_size
        if config.head_type != "linear":
            raise ValueError("S0-S3 uses the same deterministic linear state head in every variant.")

        class StateModule(nn_module.Module):
            def __init__(self):
                super().__init__()
                self.encoder = build_gru_module(context, config, torch_module, nn_module,
                    head_factory=lambda width, classes: nn_module.Identity())
                self.head = nn_module.Linear(self.encoder.embedding_size + state_size, context.num_classes)

            def forward(self, sequences, states):
                if states.shape != (len(sequences), state_size):
                    raise ValueError("Current trade states must have shape [batch, 10].")
                # Stop state/history gradients, not encoder/head gradients.
                embedding = self.encoder.encode(sequences)
                return self.head(torch_module.cat((embedding, states.detach()), dim=1))

        return StateModule()

    def fit(self, *args, **kwargs):
        raise RuntimeError("Use fit_trade_state_model with chronological execution panels.")

    def predict_proba(self, *args, **kwargs):
        raise RuntimeError("Use rollout_trade_state_model; stateless predictions are invalid here.")


def create_trade_state_gru_classifier(context, parameters):
    return create_gru_variant_classifier(context, parameters, TradeStateGRUClassifier)
