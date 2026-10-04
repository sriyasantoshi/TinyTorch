from __future__ import annotations
import numpy as np
from typing import List, Dict, Set, Iterator, Union, Sequence, Optional
from tinytorch.tensor import Tensor, ArrayLike


class Parameter(Tensor):
    """A Tensor that is automatically registered as a trainable parameter in a Module."""

    def __init__(self, data: ArrayLike, requires_grad: bool = True) -> None:
        super().__init__(data, requires_grad=requires_grad)


class Module:
    """Base class for all neural network modules."""

    def __init__(self) -> None:
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Parameter] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value: Union[Parameter, Module, object]) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> List[Parameter]:
        """Return a list of all parameters in this module and its submodules."""
        params: List[Parameter] = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self) -> None:
        """Clear the gradients of all parameters."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> Module:
        """Set the module in training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        """Set the module in evaluation mode."""
        return self.train(False)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
