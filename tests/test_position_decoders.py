import numpy as np
import pytest

from trading_system.evaluation.position_decoders import decode_positions


PROBABILITIES = np.array(
    [
        [0.60, 0.25, 0.15],
        [0.20, 0.55, 0.25],
        [0.20, 0.25, 0.55],
        [0.36, 0.30, 0.34],
    ]
)


def test_continuous_sign_and_argmax_decoders_are_distinct():
    np.testing.assert_allclose(
        decode_positions(PROBABILITIES, "continuous"),
        [-0.45, 0.05, 0.35, -0.02],
    )
    np.testing.assert_array_equal(
        decode_positions(PROBABILITIES, "sign"), [-1.0, 1.0, 1.0, -1.0]
    )
    np.testing.assert_array_equal(
        decode_positions(PROBABILITIES, "argmax"), [-1.0, 0.0, 1.0, -1.0]
    )


def test_deadband_flattens_small_directional_expectations():
    np.testing.assert_array_equal(
        decode_positions(PROBABILITIES, "deadband", threshold=0.05),
        [-1.0, 0.0, 1.0, 0.0],
    )


def test_confidence_decoder_uses_top_two_probability_margin():
    np.testing.assert_array_equal(
        decode_positions(PROBABILITIES, "confidence", threshold=0.10),
        [-1.0, 0.0, 1.0, 0.0],
    )


@pytest.mark.parametrize("decoder", ["deadband", "confidence"])
def test_threshold_decoders_require_a_valid_threshold(decoder):
    with pytest.raises(ValueError, match="requires"):
        decode_positions(PROBABILITIES, decoder)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decode_positions(PROBABILITIES, decoder, threshold=1.1)


def test_decoder_rejects_invalid_probability_contract():
    with pytest.raises(ValueError, match="sum to one"):
        decode_positions(np.ones((2, 3)), "continuous")
    with pytest.raises(ValueError, match="Unknown decoder"):
        decode_positions(PROBABILITIES, "magic")
