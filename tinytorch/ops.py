from __future__ import annotations
import numpy as np
from typing import Optional, Union, Tuple
from tinytorch.tensor import Tensor, ArrayLike, unbroadcast
from tinytorch.utils.conv_utils import (
    im2col_vectorized,
    col2im_vectorized,
    im2col_unvectorized,
    col2im_unvectorized,
    get_padding_stride,
    PaddingType,
    StrideType,
)


def add(a: ArrayLike, b: ArrayLike) -> Tensor:
    a_t = a if isinstance(a, Tensor) else Tensor(a)
    return a_t.add(b)


def mul(a: ArrayLike, b: ArrayLike) -> Tensor:
    a_t = a if isinstance(a, Tensor) else Tensor(a)
    return a_t.mul(b)


def matmul(a: ArrayLike, b: ArrayLike) -> Tensor:
    a_t = a if isinstance(a, Tensor) else Tensor(a)
    return a_t.matmul(b)


def relu(x: Tensor) -> Tensor:
    return x.relu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.softmax(axis=axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.log_softmax(axis=axis)


def sum(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    return x.sum(axis=axis, keepdims=keepdims)


def mean(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    return x.mean(axis=axis, keepdims=keepdims)


def reshape(x: Tensor, *shape: Union[int, Tuple[int, ...]]) -> Tensor:
    return x.reshape(*shape)


def flatten(x: Tensor, start_dim: int = 1) -> Tensor:
    shape = x.shape
    if start_dim == 0:
        return x.reshape(-1)
    batch_dims = shape[:start_dim]
    flat_dim = int(np.prod(shape[start_dim:]))
    return x.reshape((*batch_dims, flat_dim))


def cross_entropy(logits: Tensor, targets: Union[Tensor, np.ndarray]) -> Tensor:
    """Numerically stable Cross-Entropy Loss."""
    target_data = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
    logits_data = logits.data
    batch_size = logits_data.shape[0]

    max_logits = np.max(logits_data, axis=-1, keepdims=True)
    shifted = logits_data - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)

    if target_data.ndim == 1 or target_data.shape == (batch_size,):
        target_indices = target_data.astype(int)
        loss_val = -np.mean(log_probs[np.arange(batch_size), target_indices])
        one_hot = np.zeros_like(logits_data)
        one_hot[np.arange(batch_size), target_indices] = 1.0
    else:
        one_hot = target_data
        loss_val = -np.mean(np.sum(one_hot * log_probs, axis=-1))

    out = Tensor(loss_val, requires_grad=logits.requires_grad, children=(logits,), op="cross_entropy")

    def _backward() -> None:
        if out.grad is None or not logits.requires_grad:
            return
        g_logits = (probs - one_hot) / batch_size
        g_logits = out.grad * g_logits
        logits.grad = g_logits.copy() if logits.grad is None else logits.grad + g_logits

    out._backward = _backward
    return out


def mse_loss(input: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    target_t = target if isinstance(target, Tensor) else Tensor(target)
    diff = input.sub(target_t)
    return (diff * diff).mean()


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: StrideType = 1,
    padding: PaddingType = 0,
    vectorized: bool = True,
) -> Tensor:
    """2D Convolution operation using im2col."""
    im2col_fn = im2col_vectorized if vectorized else im2col_unvectorized
    col2im_fn = col2im_vectorized if vectorized else col2im_unvectorized

    N, C, H, W = x.shape
    C_out, C_in, k_h, k_w = weight.shape
    p_h, p_w, s_h, s_w = get_padding_stride(padding, stride)

    out_h = (H + 2 * p_h - k_h) // s_h + 1
    out_w = (W + 2 * p_w - k_w) // s_w + 1

    x_col = im2col_fn(x.data, k_h, k_w, padding=padding, stride=stride)
    w_row = weight.data.reshape(C_out, -1)

    out_col = w_row @ x_col
    if bias is not None:
        out_col += bias.data.reshape(-1, 1)

    out_data = out_col.reshape(C_out, out_h, out_w, N).transpose(3, 0, 1, 2)

    children = (x, weight) if bias is None else (x, weight, bias)
    requires_grad = x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    out = Tensor(out_data, requires_grad=requires_grad, children=children, op="conv2d")

    def _backward() -> None:
        if out.grad is None:
            return

        dout_reshaped = out.grad.transpose(1, 2, 3, 0).reshape(C_out, -1)

        if weight.requires_grad:
            dw = dout_reshaped @ x_col.T
            dw = dw.reshape(weight.shape)
            weight.grad = dw.copy() if weight.grad is None else weight.grad + dw

        if bias is not None and bias.requires_grad:
            db = np.sum(dout_reshaped, axis=1)
            bias.grad = db.copy() if bias.grad is None else bias.grad + db

        if x.requires_grad:
            dx_col = w_row.T @ dout_reshaped
            dx = col2im_fn(dx_col, x.shape, k_h, k_w, padding=padding, stride=stride)
            x.grad = dx.copy() if x.grad is None else x.grad + dx

    out._backward = _backward
    return out


def max_pool2d(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 2,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: PaddingType = 0,
) -> Tensor:
    """2D Max Pooling operation."""
    k_h, k_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    if stride is None:
        s_h, s_w = k_h, k_w
    else:
        s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding

    N, C, H, W = x.shape
    out_h = (H + 2 * p_h - k_h) // s_h + 1
    out_w = (W + 2 * p_w - k_w) // s_w + 1

    x_col = im2col_vectorized(x.data, k_h, k_w, padding=(p_h, p_w), stride=(s_h, s_w))
    # x_col shape: (C * k_h * k_w, N * out_h * out_w)
    x_col_reshaped = x_col.reshape(C, k_h * k_w, -1)

    max_idx = np.argmax(x_col_reshaped, axis=1)
    out_col = np.max(x_col_reshaped, axis=1)

    out_data = out_col.reshape(C, out_h, out_w, N).transpose(3, 0, 1, 2)
    out = Tensor(out_data, requires_grad=x.requires_grad, children=(x,), op="max_pool2d")

    def _backward() -> None:
        if out.grad is None or not x.requires_grad:
            return

        dout_col = out.grad.transpose(1, 2, 3, 0).reshape(C, -1)
        dx_col_reshaped = np.zeros_like(x_col_reshaped)

        # Set max indices
        c_idx = np.arange(C)[:, None]
        spatial_idx = np.arange(dout_col.shape[1])[None, :]
        dx_col_reshaped[c_idx, max_idx, spatial_idx] = dout_col

        dx_col = dx_col_reshaped.reshape(C * k_h * k_w, -1)
        dx = col2im_vectorized(dx_col, x.shape, k_h, k_w, padding=(p_h, p_w), stride=(s_h, s_w))
        x.grad = dx.copy() if x.grad is None else x.grad + dx

    out._backward = _backward
    return out
