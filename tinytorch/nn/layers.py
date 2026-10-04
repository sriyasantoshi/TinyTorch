from __future__ import annotations
import numpy as np
from typing import Optional, Union, Tuple
from tinytorch.tensor import Tensor
from tinytorch.nn.module import Module, Parameter
import tinytorch.ops as ops
from tinytorch.utils.conv_utils import PaddingType, StrideType


class Linear(Module):
    """Applies a linear transformation: y = x @ weight + bias."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # He / Kaiming normal initialization
        bound = np.sqrt(2.0 / in_features)
        w_data = np.random.randn(in_features, out_features).astype(np.float32) * bound
        self.weight = Parameter(w_data)

        if bias:
            b_data = np.zeros(out_features, dtype=np.float32)
            self.bias: Optional[Parameter] = Parameter(b_data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out.add(self.bias)
        return out


class Conv2d(Module):
    """2D Convolutional layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: StrideType = 1,
        padding: PaddingType = 0,
        bias: bool = True,
        vectorized: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.vectorized = vectorized

        k_h, k_w = self.kernel_size
        bound = np.sqrt(2.0 / (in_channels * k_h * k_w))
        w_data = np.random.randn(out_channels, in_channels, k_h, k_w).astype(np.float32) * bound
        self.weight = Parameter(w_data)

        if bias:
            b_data = np.zeros(out_channels, dtype=np.float32)
            self.bias: Optional[Parameter] = Parameter(b_data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return ops.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            vectorized=self.vectorized,
        )


class Flatten(Module):
    """Flattens input tensor along specified start dimension."""

    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        return ops.flatten(x, start_dim=self.start_dim)


class MaxPool2d(Module):
    """2D Max Pooling layer."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        stride: Optional[StrideType] = None,
        padding: PaddingType = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return ops.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
