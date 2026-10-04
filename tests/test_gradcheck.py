import pytest
import numpy as np
from tinytorch.tensor import Tensor
import tinytorch.ops as ops


def numerical_gradcheck(func, inputs, eps=1e-4, atol=1e-3, rtol=1e-3):
    """Compares analytical autograd gradients with numerical finite differences."""
    # Convert inputs to float64 for high-precision finite differences
    for inp in inputs:
        inp.data = inp.data.astype(np.float64)
        inp.zero_grad()
    
    out = func(*inputs)
    out.backward()

    for idx, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue
        
        analytical = inp.grad.copy()
        num_grad = np.zeros_like(inp.data)

        # Finite difference loop over all elements of tensor
        it = np.nditer(inp.data, flags=["multi_index"])
        while not it.finished:
            multi_idx = it.multi_index
            orig_val = float(inp.data[multi_idx])

            # f(x + eps)
            inp.data[multi_idx] = orig_val + eps
            out_pos = float(func(*inputs).data)

            # f(x - eps)
            inp.data[multi_idx] = orig_val - eps
            out_neg = float(func(*inputs).data)

            # Restore original value
            inp.data[multi_idx] = orig_val

            num_grad[multi_idx] = (out_pos - out_neg) / (2.0 * eps)
            it.iternext()

        np.testing.assert_allclose(
            analytical,
            num_grad,
            atol=atol,
            rtol=rtol,
            err_msg=f"Gradcheck failed for input index {idx}",
        )


def test_gradcheck_add():
    x = Tensor([[1.5, -2.0], [0.5, 3.0]], requires_grad=True)
    y = Tensor([[0.2, 1.1], [-0.5, 0.4]], requires_grad=True)
    func = lambda a, b: (ops.add(a, b) * 2.0).sum()
    numerical_gradcheck(func, [x, y])


def test_gradcheck_mul():
    x = Tensor([[1.5, -2.0], [0.5, 3.0]], requires_grad=True)
    y = Tensor([[0.2, 1.1], [-0.5, 0.4]], requires_grad=True)
    func = lambda a, b: ops.mul(a, b).sum()
    numerical_gradcheck(func, [x, y])


def test_gradcheck_matmul():
    x = Tensor([[1.5, -2.0, 0.1], [0.5, 3.0, -1.2]], requires_grad=True)
    w = Tensor([[0.2, 1.1], [-0.5, 0.4], [1.0, -0.8]], requires_grad=True)
    func = lambda a, b: ops.matmul(a, b).sum()
    numerical_gradcheck(func, [x, w])


def test_gradcheck_relu():
    x = Tensor([[-1.5, 2.0], [0.5, -3.0]], requires_grad=True)
    func = lambda a: ops.relu(a).sum()
    numerical_gradcheck(func, [x])


def test_gradcheck_tanh():
    x = Tensor([[-1.5, 2.0], [0.5, -0.3]], requires_grad=True)
    func = lambda a: ops.tanh(a).sum()
    numerical_gradcheck(func, [x])


def test_gradcheck_softmax():
    x = Tensor([[1.5, -2.0, 0.5], [0.5, 3.0, -1.2]], requires_grad=True)
    func = lambda a: (ops.softmax(a, axis=-1) * Tensor([[1.0, 0.5, 0.2], [0.1, 0.2, 0.3]])).sum()
    numerical_gradcheck(func, [x])


def test_gradcheck_cross_entropy():
    logits = Tensor([[1.5, -2.0, 0.5], [0.5, 3.0, -1.2]], requires_grad=True)
    targets = np.array([0, 2])
    func = lambda l: ops.cross_entropy(l, targets)
    numerical_gradcheck(func, [logits])


def test_gradcheck_reshape():
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    func = lambda a: ops.reshape(a, (2, 3)).sum()
    numerical_gradcheck(func, [x])


def test_gradcheck_conv2d():
    x = Tensor(np.random.randn(2, 1, 5, 5), requires_grad=True)
    w = Tensor(np.random.randn(2, 1, 3, 3), requires_grad=True)
    b = Tensor(np.random.randn(2), requires_grad=True)
    func = lambda inp, weight, bias: ops.conv2d(inp, weight, bias, stride=1, padding=1).sum()
    numerical_gradcheck(func, [x, w, b])


def test_gradcheck_max_pool2d():
    x = Tensor(np.random.randn(2, 2, 4, 4), requires_grad=True)
    func = lambda inp: ops.max_pool2d(inp, kernel_size=2, stride=2).sum()
    numerical_gradcheck(func, [x])
