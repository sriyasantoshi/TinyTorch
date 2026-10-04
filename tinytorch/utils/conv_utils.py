import numpy as np
from typing import Tuple, Union

PaddingType = Union[int, Tuple[int, int]]
StrideType = Union[int, Tuple[int, int]]


def get_padding_stride(padding: PaddingType, stride: StrideType) -> Tuple[int, int, int, int]:
    p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
    s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    return p_h, p_w, s_h, s_w


def im2col_unvectorized(
    x: np.ndarray,
    k_h: int,
    k_w: int,
    padding: PaddingType = 0,
    stride: StrideType = 1,
) -> np.ndarray:
    """Unvectorized im2col using nested Python loops.
    
    Args:
        x: Input array of shape (N, C, H, W).
        k_h: Kernel height.
        k_w: Kernel width.
        padding: Padding size.
        stride: Stride size.

    Returns:
        np.ndarray of shape (C * k_h * k_w, N * out_h * out_w).
    """
    p_h, p_w, s_h, s_w = get_padding_stride(padding, stride)
    N, C, H, W = x.shape
    out_h = (H + 2 * p_h - k_h) // s_h + 1
    out_w = (W + 2 * p_w - k_w) // s_w + 1

    x_padded = np.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode="constant")
    cols = np.zeros((C * k_h * k_w, N * out_h * out_w), dtype=x.dtype)

    col_idx = 0
    for n in range(N):
        for h in range(out_h):
            for w in range(out_w):
                h_start = h * s_h
                w_start = w * s_w
                # Extract window
                window = x_padded[n, :, h_start : h_start + k_h, w_start : w_start + k_w]
                cols[:, col_idx] = window.reshape(-1)
                col_idx += 1

    return cols


def col2im_unvectorized(
    cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    k_h: int,
    k_w: int,
    padding: PaddingType = 0,
    stride: StrideType = 1,
) -> np.ndarray:
    """Unvectorized col2im using nested Python loops."""
    p_h, p_w, s_h, s_w = get_padding_stride(padding, stride)
    N, C, H, W = x_shape
    out_h = (H + 2 * p_h - k_h) // s_h + 1
    out_w = (W + 2 * p_w - k_w) // s_w + 1

    H_padded, W_padded = H + 2 * p_h, W + 2 * p_w
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    col_idx = 0
    for n in range(N):
        for h in range(out_h):
            for w in range(out_w):
                h_start = h * s_h
                w_start = w * s_w
                window = cols[:, col_idx].reshape(C, k_h, k_w)
                x_padded[n, :, h_start : h_start + k_h, w_start : w_start + k_w] += window
                col_idx += 1

    if p_h == 0 and p_w == 0:
        return x_padded
    return x_padded[:, :, p_h : p_h + H, p_w : p_w + W]


def get_im2col_indices(
    x_shape: Tuple[int, int, int, int],
    k_h: int,
    k_w: int,
    padding: PaddingType = 0,
    stride: StrideType = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_h, p_w, s_h, s_w = get_padding_stride(padding, stride)
    N, C, H, W = x_shape

    out_h = (H + 2 * p_h - k_h) // s_h + 1
    out_w = (W + 2 * p_w - k_w) // s_w + 1

    i0 = np.repeat(np.arange(k_h), k_w)
    i0 = np.tile(i0, C)
    i1 = s_h * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(k_w), k_h * C)
    j1 = s_w * np.tile(np.arange(out_w), out_h)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), k_h * k_w).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int)


def im2col_vectorized(
    x: np.ndarray,
    k_h: int,
    k_w: int,
    padding: PaddingType = 0,
    stride: StrideType = 1,
) -> np.ndarray:
    """Fully vectorized im2col using advanced indexing."""
    p_h, p_w, _, _ = get_padding_stride(padding, stride)
    N, C, H, W = x.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode="constant")

    k, i, j = get_im2col_indices(x.shape, k_h, k_w, padding, stride)
    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(C * k_h * k_w, -1)
    return cols


def col2im_vectorized(
    cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    k_h: int,
    k_w: int,
    padding: PaddingType = 0,
    stride: StrideType = 1,
) -> np.ndarray:
    """Fully vectorized col2im using np.add.at."""
    p_h, p_w, _, _ = get_padding_stride(padding, stride)
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * p_h, W + 2 * p_w
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j = get_im2col_indices(x_shape, k_h, k_w, padding, stride)
    cols_reshaped = cols.reshape(C * k_h * k_w, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if p_h == 0 and p_w == 0:
        return x_padded
    return x_padded[:, :, p_h : p_h + H, p_w : p_w + W]
