from __future__ import annotations
from tinytorch.tensor import Tensor
from tinytorch.nn.module import Module
import tinytorch.ops as ops


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)


class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return ops.softmax(x, axis=self.axis)


class LogSoftmax(Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return ops.log_softmax(x, axis=self.axis)
