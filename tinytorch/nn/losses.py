from __future__ import annotations
import numpy as np
from typing import Union
from tinytorch.tensor import Tensor
from tinytorch.nn.module import Module
import tinytorch.ops as ops


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, targets: Union[Tensor, np.ndarray]) -> Tensor:
        return ops.cross_entropy(logits, targets)


class MSELoss(Module):
    def forward(self, input: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
        return ops.mse_loss(input, target)
