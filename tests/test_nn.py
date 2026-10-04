import pytest
import numpy as np
from tinytorch.tensor import Tensor
import tinytorch.nn as nn
import tinytorch.optim as optim


def test_sequential_forward():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    x = Tensor(np.random.randn(4, 10), requires_grad=True)
    out = model(x)
    assert out.shape == (4, 5)
    assert len(model.parameters()) == 4


def test_sgd_optimizer():
    w = nn.Parameter(np.array([2.0, 3.0]))
    w.grad = np.array([0.1, -0.2])
    optimizer = optim.SGD([w], lr=0.1, momentum=0.9)
    optimizer.step()

    # v = 0.9 * 0 + grad = [0.1, -0.2]
    # w = [2.0, 3.0] - 0.1 * [0.1, -0.2] = [1.99, 3.02]
    np.testing.assert_allclose(w.data, [1.99, 3.02])


def test_adam_optimizer():
    w = nn.Parameter(np.array([2.0, 3.0]))
    w.grad = np.array([0.1, -0.2])
    optimizer = optim.Adam([w], lr=0.1)
    optimizer.step()
    assert w.data[0] < 2.0
    assert w.data[1] > 3.0


def test_lr_schedulers():
    w = nn.Parameter(np.array([1.0]))
    optimizer = optim.SGD([w], lr=0.1)
    scheduler = optim.StepLR(optimizer, step_size=2, gamma=0.5)

    assert optimizer.lr == 0.1  # Epoch 0
    scheduler.step()
    assert optimizer.lr == 0.1  # Epoch 1
    scheduler.step()
    assert abs(optimizer.lr - 0.05) < 1e-6  # Epoch 2
