from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Sequence
from tinytorch.nn.module import Parameter


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params: Sequence[Parameter], lr: float = 1e-3) -> None:
        self.params: List[Parameter] = list(params)
        self.lr: float = lr
        self.state: Dict[Parameter, Dict[str, np.ndarray]] = {}

    def zero_grad(self) -> None:
        """Zero out gradients for all tracked parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum and weight decay."""

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.copy()
            if self.weight_decay != 0.0:
                grad += self.weight_decay * p.data

            if self.momentum != 0.0:
                if p not in self.state:
                    self.state[p] = {"v": np.zeros_like(p.data)}
                v = self.state[p]["v"]
                v = self.momentum * v + grad
                self.state[p]["v"] = v
                p.data -= self.lr * v
            else:
                p.data -= self.lr * grad


class Adam(Optimizer):
    """Adam optimizer with bias correction and weight decay."""

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t: int = 0

    def step(self) -> None:
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.copy()
            if self.weight_decay != 0.0:
                grad += self.weight_decay * p.data

            if p not in self.state:
                self.state[p] = {
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                }

            m = self.state[p]["m"]
            v = self.state[p]["v"]

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

            self.state[p]["m"] = m
            self.state[p]["v"] = v

            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
