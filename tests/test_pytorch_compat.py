import pytest
import numpy as np
import torch
from tinytorch.tensor import Tensor
import tinytorch.ops as ops
import tinytorch.nn as nn


def test_pytorch_compat_linear():
    np.random.seed(42)
    torch.manual_seed(42)

    x_np = np.random.randn(4, 8).astype(np.float32)
    w_np = np.random.randn(8, 16).astype(np.float32)
    b_np = np.random.randn(16).astype(np.float32)

    # PyTorch setup
    tx = torch.tensor(x_np, requires_grad=True)
    tw = torch.tensor(w_np, requires_grad=True)
    tb = torch.tensor(b_np, requires_grad=True)
    tout = (tx @ tw + tb).relu().sum()
    tout.backward()

    # TinyTorch setup
    ttx = Tensor(x_np, requires_grad=True)
    ttw = Tensor(w_np, requires_grad=True)
    ttb = Tensor(b_np, requires_grad=True)
    ttout = (ttx.matmul(ttw) + ttb).relu().sum()
    ttout.backward()

    # Compare forward
    np.testing.assert_allclose(ttout.data, tout.detach().numpy(), atol=1e-5)

    # Compare gradients
    np.testing.assert_allclose(ttx.grad, tx.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(ttw.grad, tw.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(ttb.grad, tb.grad.numpy(), atol=1e-5)


def test_pytorch_compat_cross_entropy():
    np.random.seed(42)

    logits_np = np.random.randn(5, 10).astype(np.float32)
    targets_np = np.array([1, 0, 4, 3, 9], dtype=np.int64)

    # PyTorch
    tx = torch.tensor(logits_np, requires_grad=True)
    ttargets = torch.tensor(targets_np, dtype=torch.long)
    tloss = torch.nn.functional.cross_entropy(tx, ttargets)
    tloss.backward()

    # TinyTorch
    ttx = Tensor(logits_np, requires_grad=True)
    ttloss = ops.cross_entropy(ttx, targets_np)
    ttloss.backward()

    np.testing.assert_allclose(ttloss.data, tloss.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(ttx.grad, tx.grad.numpy(), atol=1e-5)


def test_pytorch_compat_conv2d():
    np.random.seed(42)

    x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    w_np = np.random.randn(4, 3, 3, 3).astype(np.float32)
    b_np = np.random.randn(4).astype(np.float32)

    # PyTorch
    tx = torch.tensor(x_np, requires_grad=True)
    tw = torch.tensor(w_np, requires_grad=True)
    tb = torch.tensor(b_np, requires_grad=True)
    tout = torch.nn.functional.conv2d(tx, tw, bias=tb, stride=1, padding=1).sum()
    tout.backward()

    # TinyTorch
    ttx = Tensor(x_np, requires_grad=True)
    ttw = Tensor(w_np, requires_grad=True)
    ttb = Tensor(b_np, requires_grad=True)
    ttout = ops.conv2d(ttx, ttw, bias=ttb, stride=1, padding=1).sum()
    ttout.backward()

    np.testing.assert_allclose(ttout.data, tout.detach().numpy(), atol=1e-4)
    np.testing.assert_allclose(ttx.grad, tx.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(ttw.grad, tw.grad.numpy(), atol=1e-4)
    np.testing.assert_allclose(ttb.grad, tb.grad.numpy(), atol=1e-4)
