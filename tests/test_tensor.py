import pytest
import numpy as np
from tinytorch.tensor import Tensor
import tinytorch.ops as ops


def test_tensor_creation():
    t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    assert t.shape == (3,)
    assert t.requires_grad is True
    assert t.grad is None


def test_tensor_add_autograd():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0], requires_grad=True)
    z = (x + y) * Tensor([2.0, 3.0], requires_grad=False)
    out = z.sum()
    out.backward()

    np.testing.assert_allclose(x.grad, [2.0, 3.0])
    np.testing.assert_allclose(y.grad, [2.0, 3.0])


def test_tensor_matmul_autograd():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w = Tensor([[0.5, -0.5], [1.0, 2.0]], requires_grad=True)
    out = (x @ w).sum()
    out.backward()

    # d(sum(X @ W))/dX = W.T @ ones = W @ ones transposed
    expected_x_grad = np.ones((2, 2)) @ w.data.T
    expected_w_grad = x.data.T @ np.ones((2, 2))

    np.testing.assert_allclose(x.grad, expected_x_grad)
    np.testing.assert_allclose(w.grad, expected_w_grad)


def test_tensor_zero_grad():
    x = Tensor([2.0], requires_grad=True)
    y = x * 3.0
    y.backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None


def test_relu_tanh_sigmoid():
    x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    r = ops.relu(x).sum()
    r.backward()
    np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0])

    x.zero_grad()
    t = ops.tanh(x).sum()
    t.backward()
    expected_tanh_grad = 1.0 - np.tanh([-1.0, 0.0, 2.0]) ** 2
    np.testing.assert_allclose(x.grad, expected_tanh_grad, atol=1e-5)
