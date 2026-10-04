from __future__ import annotations
import numpy as np
from typing import Optional, Set, Tuple, Union, List, Callable, Sequence

ArrayLike = Union[float, int, Sequence, np.ndarray, 'Tensor']


def unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Sum out dimensions added or expanded by broadcasting.
    
    Args:
        grad: The incoming gradient array.
        target_shape: The shape of the original tensor to reduce to.

    Returns:
        np.ndarray: Gradient array matching `target_shape`.
    """
    if grad.shape == target_shape:
        return grad
    if target_shape == ():
        return np.sum(grad)

    # 1. Sum over extra leading dimensions
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = np.sum(grad, axis=tuple(range(ndim_diff)))

    # 2. Sum over broadcasted dimensions where target has size 1
    axes_to_sum = [i for i, dim in enumerate(target_shape) if dim == 1 and grad.shape[i] > 1]
    if axes_to_sum:
        grad = np.sum(grad, axis=tuple(axes_to_sum), keepdims=True)

    return grad.reshape(target_shape)


class Tensor:
    """TinyTorch Tensor with reverse-mode automatic differentiation."""

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        children: Tuple[Tensor, ...] = (),
        op: str = "",
    ) -> None:
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        elif isinstance(data, (np.ndarray, np.number)):
            arr = np.asarray(data)
            dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float32
            self.data = arr.astype(dtype, copy=False)
        else:
            self.data = np.array(data, dtype=np.float32)

        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._prev: Set[Tensor] = set(children)
        self._op: str = op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def zero_grad(self) -> None:
        """Reset gradient to None."""
        self.grad = None

    def detach(self) -> Tensor:
        """Return a new Tensor detached from the computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)

    def backward(self, gradient: Optional[Union[np.ndarray, Tensor]] = None) -> None:
        """Run reverse-mode automatic differentiation.
        
        Args:
            gradient: Incoming gradient array/tensor. Defaults to 1.0 for scalar/root outputs.
        """
        if not self.requires_grad:
            return

        if gradient is None:
            if self.data.ndim == 0 or self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                gradient = np.ones_like(self.data)
        elif isinstance(gradient, Tensor):
            gradient = gradient.data
        elif not isinstance(gradient, np.ndarray):
            gradient = np.array(gradient, dtype=self.data.dtype)

        if self.grad is None:
            self.grad = gradient.copy()
        else:
            self.grad = self.grad + gradient

        # Build topological sorting
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Execute backward pass in reverse topological order
        for node in reversed(topo):
            node._backward()

    # --- Basic Operations ---

    def add(self, other: ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            children=(self, other_t),
            op="+",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                g_self = unbroadcast(out.grad, self.shape)
                self.grad = g_self.copy() if self.grad is None else self.grad + g_self
            if other_t.requires_grad:
                g_other = unbroadcast(out.grad, other_t.shape)
                other_t.grad = g_other.copy() if other_t.grad is None else other_t.grad + g_other

        out._backward = _backward
        return out

    def sub(self, other: ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data - other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            children=(self, other_t),
            op="-",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                g_self = unbroadcast(out.grad, self.shape)
                self.grad = g_self.copy() if self.grad is None else self.grad + g_self
            if other_t.requires_grad:
                g_other = unbroadcast(-out.grad, other_t.shape)
                other_t.grad = g_other.copy() if other_t.grad is None else other_t.grad + g_other

        out._backward = _backward
        return out

    def mul(self, other: ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            children=(self, other_t),
            op="*",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                g_self = unbroadcast(out.grad * other_t.data, self.shape)
                self.grad = g_self.copy() if self.grad is None else self.grad + g_self
            if other_t.requires_grad:
                g_other = unbroadcast(out.grad * self.data, other_t.shape)
                other_t.grad = g_other.copy() if other_t.grad is None else other_t.grad + g_other

        out._backward = _backward
        return out

    def div(self, other: ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            children=(self, other_t),
            op="/",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                g_self = unbroadcast(out.grad / other_t.data, self.shape)
                self.grad = g_self.copy() if self.grad is None else self.grad + g_self
            if other_t.requires_grad:
                g_other = unbroadcast(-out.grad * self.data / (other_t.data ** 2), other_t.shape)
                other_t.grad = g_other.copy() if other_t.grad is None else other_t.grad + g_other

        out._backward = _backward
        return out

    def matmul(self, other: ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out_data = np.matmul(self.data, other_t.data)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            children=(self, other_t),
            op="@",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                # Transpose last two axes of other_t for batched/2D matmul grad
                if other_t.data.ndim == 1:
                    # 1D vector case
                    g_self = np.outer(out.grad, other_t.data)
                elif self.data.ndim == 1:
                    g_self = np.matmul(out.grad, other_t.data.T)
                else:
                    g_self = np.matmul(out.grad, np.swapaxes(other_t.data, -1, -2))
                g_self = unbroadcast(g_self, self.shape)
                self.grad = g_self.copy() if self.grad is None else self.grad + g_self

            if other_t.requires_grad:
                if self.data.ndim == 1:
                    g_other = np.outer(self.data, out.grad)
                elif other_t.data.ndim == 1:
                    g_other = np.matmul(self.data.T, out.grad)
                else:
                    g_other = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                g_other = unbroadcast(g_other, other_t.shape)
                other_t.grad = g_other.copy() if other_t.grad is None else other_t.grad + g_other

        out._backward = _backward
        return out

    def pow(self, power: Union[float, int, Tensor]) -> Tensor:
        if isinstance(power, Tensor):
            out_data = self.data ** power.data
            out = Tensor(
                out_data,
                requires_grad=self.requires_grad or power.requires_grad,
                children=(self, power),
                op="**",
            )

            def _backward() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    g_self = unbroadcast(out.grad * power.data * (self.data ** (power.data - 1)), self.shape)
                    self.grad = g_self.copy() if self.grad is None else self.grad + g_self
                if power.requires_grad:
                    g_power = unbroadcast(out.grad * out_data * np.log(np.maximum(self.data, 1e-12)), power.shape)
                    power.grad = g_power.copy() if power.grad is None else power.grad + g_power

            out._backward = _backward
            return out
        else:
            out_data = self.data ** power
            out = Tensor(
                out_data,
                requires_grad=self.requires_grad,
                children=(self,),
                op=f"**{power}",
            )

            def _backward() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    g_self = out.grad * power * (self.data ** (power - 1))
                    self.grad = g_self.copy() if self.grad is None else self.grad + g_self

            out._backward = _backward
            return out

    def neg(self) -> Tensor:
        return self.mul(-1.0)

    # --- Reductions & Shape Operations ---

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op="sum")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                # Re-expand reduced axes for broadcasting
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = axis
                # Handle negative axes
                axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                shape = list(self.shape)
                for a in axes:
                    shape[a] = 1
                grad = grad.reshape(shape)
            g_self = np.broadcast_to(grad, self.shape)
            self.grad = g_self.copy() if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op="mean")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            num_elements = self.size / out_data.size if out_data.size > 0 else 1.0
            grad = out.grad / num_elements
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                shape = list(self.shape)
                for a in axes:
                    shape[a] = 1
                grad = grad.reshape(shape)
            g_self = np.broadcast_to(grad, self.shape)
            self.grad = g_self.copy() if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(shape)  # type: ignore

        out_data = np.reshape(self.data, target_shape)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op="reshape")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            g_self = out.grad.reshape(self.shape)
            self.grad = g_self.copy() if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    def transpose(self, *axes: Union[int, Tuple[int, ...]]) -> Tensor:
        if len(axes) == 0:
            perm = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
        else:
            perm = tuple(axes)  # type: ignore

        out_data = np.transpose(self.data, perm)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op="transpose")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            if perm is None:
                g_self = np.transpose(out.grad)
            else:
                inv_perm = np.argsort(perm)
                g_self = np.transpose(out.grad, inv_perm)
            self.grad = g_self.copy() if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    @property
    def T(self) -> Tensor:
        return self.transpose()

    # --- Activation Ops ---

    def relu(self) -> Tensor:
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op="relu")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            g_self = out.grad * (self.data > 0)
            self.grad = g_self.copy() if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        out_data = np.tanh(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op="tanh")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            g_self = out.grad * (1.0 - out_data ** 2)
            self.grad = g_self.copy() if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        out_data = 1.0 / (1.0 + np.exp(-np.clip(self.data, -50, 50)))
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op="sigmoid")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            g_self = out.grad * out_data * (1.0 - out_data)
            self.grad = g_self.copy() if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> Tensor:
        max_val = np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(self.data - max_val)
        out_data = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op="softmax")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            # dL/dx = S * (dL/dS - sum(dL/dS * S, keepdims=True))
            sum_grad_s = np.sum(out.grad * out_data, axis=axis, keepdims=True)
            g_self = (out.grad - sum_grad_s) * out_data
            self.grad = g_self.copy() if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> Tensor:
        max_val = np.max(self.data, axis=axis, keepdims=True)
        shifted = self.data - max_val
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        out_data = shifted - log_sum_exp
        out = Tensor(out_data, requires_grad=self.requires_grad, children=(self,), op="log_softmax")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            softmax_vals = np.exp(out_data)
            sum_grad = np.sum(out.grad, axis=axis, keepdims=True)
            g_self = out.grad - sum_grad * softmax_vals
            self.grad = g_self.copy() if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    # --- Operator Overloads ---

    def __add__(self, other: ArrayLike) -> Tensor:
        return self.add(other)

    def __radd__(self, other: ArrayLike) -> Tensor:
        return self.add(other)

    def __sub__(self, other: ArrayLike) -> Tensor:
        return self.sub(other)

    def __rsub__(self, other: ArrayLike) -> Tensor:
        return Tensor(other).sub(self)

    def __mul__(self, other: ArrayLike) -> Tensor:
        return self.mul(other)

    def __rmul__(self, other: ArrayLike) -> Tensor:
        return self.mul(other)

    def __truediv__(self, other: ArrayLike) -> Tensor:
        return self.div(other)

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        return Tensor(other).div(self)

    def __matmul__(self, other: ArrayLike) -> Tensor:
        return self.matmul(other)

    def __rmatmul__(self, other: ArrayLike) -> Tensor:
        return Tensor(other).matmul(self)

    def __pow__(self, power: Union[float, int, Tensor]) -> Tensor:
        return self.pow(power)

    def __neg__(self) -> Tensor:
        return self.neg()

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
