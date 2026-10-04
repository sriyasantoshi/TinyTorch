from __future__ import annotations
import numpy as np
from tinytorch.optim.optimizer import Optimizer


class _LRScheduler:
    """Base learning rate scheduler."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch
        self.step()

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self) -> None:
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.lr = new_lr


class StepLR(_LRScheduler):
    """Decays learning rate by gamma every step_size epochs."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self.optimizer.lr
        return self.optimizer.lr * self.gamma


class CosineAnnealingLR(_LRScheduler):
    """Cosine annealing learning rate scheduler."""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch == 0:
            return self.base_lr
        return float(
            self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * self.last_epoch / self.T_max))
        )
