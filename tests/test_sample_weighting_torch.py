import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trading_system.models.neural.trainer import (
    _batch_loss, build_tensor_loader, evaluate_loss, fit_torch_model,
)
from trading_system.models.neural.config import CommonTrainingConfig


def test_torch_joint_loss_and_gradient_match_formula():
    logits = torch.tensor([[1., 0., -1.], [0., 1., 2.]], requires_grad=True)
    labels = torch.tensor([0, 1])
    sample = torch.tensor([1., 4.])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([2., 3., 1.]))
    loss, mass = _batch_loss(logits, labels, criterion, sample, torch)
    expected = -(torch.log_softmax(logits, 1)[range(2), labels] * torch.tensor([2., 12.])).sum() / 14
    torch.testing.assert_close(loss, expected)
    assert mass == 14
    gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(expected, logits)[0])


def test_loader_keeps_weights_aligned_when_shuffling():
    X = np.arange(12, dtype=np.float32).reshape(12, 1, 1)
    weights = np.arange(12, dtype=np.float32) + 1
    loader = build_tensor_loader(X, np.arange(12) % 3, sample_weight=weights,
                                batch_size=4, shuffle=True, seed=3, num_workers=0, torch_module=torch)
    for features, labels, sample in loader:
        indices = features[:, 0, 0].long()
        torch.testing.assert_close(sample, indices.float() + 1)
        torch.testing.assert_close(labels, indices % 3)


def test_validation_loss_is_invariant_to_batch_partition_with_sample_weights():
    X = np.arange(12, dtype=np.float32).reshape(4, 1, 3) / 10
    y = np.array([0, 1, 2, 1])
    sample = np.array([0, 1, 2, 4], dtype=np.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([2., 3., 1.]))
    model = torch.nn.Flatten(start_dim=1)
    values = []
    for size in (1, 2, 4):
        loader = build_tensor_loader(X, y, sample_weight=sample, batch_size=size, shuffle=False,
                                    seed=3, num_workers=0, torch_module=torch)
        values.append(evaluate_loss(model, loader, criterion, "cpu", torch_module=torch))
    np.testing.assert_allclose(values, values[0], rtol=1e-6)


def test_torch_training_with_zero_weight_batches():
    X = np.random.default_rng(2).normal(size=(6, 1, 3)).astype(np.float32)
    y = np.arange(6) % 3
    sample = np.array([0, 0, 0, 1, 2, 3], dtype=np.float32)
    model = torch.nn.Sequential(torch.nn.Flatten(start_dim=1), torch.nn.Linear(3, 3))
    result = fit_torch_model(model, X, y, X, y, num_classes=3,
                             config=CommonTrainingConfig(epochs=1, batch_size=1, device="cpu"),
                             sample_weight=sample, sample_weight_val=sample)
    assert np.isfinite(result.history.val_loss).all()
