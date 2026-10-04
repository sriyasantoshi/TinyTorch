from __future__ import annotations
from typing import Sequence
from tinytorch.tensor import Tensor
from tinytorch.nn.module import Module


class Sequential(Module):
    """A sequential container for modules."""

    def __init__(self, *args: Module) -> None:
        super().__init__()
        for i, module in enumerate(args):
            setattr(self, str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x
