# TinyTorch

TinyTorch is a small learning project: a neural-network library written in Python and NumPy. It includes a reverse-mode automatic differentiation engine, tensor operations, a compact `nn` API, optimizers, and tests for selected gradients and operations.

---

## Features

1. **Reverse-mode automatic differentiation**: Builds a dynamic computation graph and accumulates gradients during backpropagation.
2. **Broadcasting gradients**: Reduces gradients across broadcast dimensions to match each input tensor's shape.
3. **Tensor operations**: Includes arithmetic, matrix multiplication, activations, reductions, shape operations, `conv2d` (via im2col), `max_pool2d`, and `flatten`.
4. **`nn` modules and optimizers**:
   - Layers: `Linear`, `Conv2d`, `Flatten`, `MaxPool2d`
   - Activations: `ReLU`, `Tanh`, `Sigmoid`, `Softmax`, `LogSoftmax`
   - Containers & Losses: `Sequential`, `MSELoss`, `CrossEntropyLoss`
   - Optimizers: `SGD` (with momentum and weight decay), `Adam` (with bias correction and weight decay)
   - LR Schedulers: `StepLR`, `CosineAnnealingLR`
5. **Tests**: Finite-difference checks for selected operations, Hypothesis tests for broadcastable shapes, and comparisons with PyTorch for a few operations.
6. **Convolution profiling**: Includes vectorized and loop-based `im2col` / `col2im` implementations and a small script to compare their training-step times.

---

## Implementation Notes

### 1. Reverse-Mode Autograd Engine

Each `Tensor` object wraps a NumPy array (`self.data`), tracks whether it requires gradients (`self.requires_grad`), maintains accumulated gradients (`self.grad`), and stores references to parent nodes (`self._prev`) along with a backward VJP closure (`self._backward`).

When `.backward()` is called:
1. A topological order of all nodes in the computation graph is constructed using depth-first search.
2. The root gradient is initialized (defaulting to 1.0 for scalars).
3. Traversal runs in **reverse topological order**, executing each node's `_backward()` closure to compute vector-Jacobian products (VJPs) and accumulate gradients into parent tensors.

### 2. Handling Broadcasting in Reverse Pass

When operating on tensors of unequal shapes (e.g. $(3, 4)$ and $(4,)$), NumPy automatically broadcasts dimensions. In the backward pass, the output gradient has shape $(3, 4)$, which must be reduced back to the input tensor shape $(4,)$.

TinyTorch enforces strict broadcast gradient reduction via `unbroadcast(grad, target_shape)`:
$$\text{unbroadcast}(\mathbf{G}, \mathbf{S}_{\text{target}}) = \text{reshape}\left( \sum_{\text{axes with } S_i = 1} \sum_{\text{extra leading axes}} \mathbf{G}, \mathbf{S}_{\text{target}} \right)$$

1. **Extra Leading Axes**: Sum out leading dimensions added by broadcasting ($N_{\text{grad}} - N_{\text{target}}$).
2. **Dimension Matching**: For dimensions where `target_shape[i] == 1` and `grad.shape[i] > 1`, sum along axis `i` with `keepdims=True`.

### 3. Vectorized Convolution via `im2col`

Conv2d converts 2D spatial convolution into matrix multiplication:
1. **Forward Pass**: Extract input patches into columns matrix $X_{\text{col}}$ of shape $(C_{\text{in}} \cdot K_h \cdot K_w, N \cdot H_{\text{out}} \cdot W_{\text{out}})$. Flatten kernel weights into rows $W_{\text{row}}$ of shape $(C_{\text{out}}, C_{\text{in}} \cdot K_h \cdot K_w)$. Compute matrix product $Y_{\text{col}} = W_{\text{row}} \cdot X_{\text{col}} + b$.
2. **Backward Pass**:
   - $\frac{\partial L}{\partial W} = \mathbf{d}Y_{\text{col}} \cdot X_{\text{col}}^T$
   - $\frac{\partial L}{\partial b} = \sum_{\text{cols}} \mathbf{d}Y_{\text{col}}$
   - $\frac{\partial L}{\partial X} = \text{col2im}\left( W_{\text{row}}^T \cdot \mathbf{d}Y_{\text{col}} \right)$

---

## Convolution Timing

One local run of `examples/profile_conv.py` measured the following for a synthetic batch of 16 single-channel $14 \times 14$ images. Each timed step includes a convolution and linear layer, loss calculation, backward pass, and optimizer update; the script averages three runs.

| Implementation Mode | Time per Step | Speedup |
| :--- | :--- | :--- |
| **Unvectorized (Nested Python loops)** | `6.19 ms` | $1.0\times$ (Baseline) |
| **Vectorized (`im2col` + `np.add.at`)** | `1.02 ms` | **$6.06\times$ Faster** |

These are measurements from that run, not a general performance guarantee; results depend on the machine and NumPy build.

---

## MNIST Examples

The example scripts report test accuracy after training. The values below are from recorded runs and can vary between runs and depending on whether OpenML MNIST could be downloaded (the loader falls back to scikit-learn's digits dataset if it cannot).

### MLP
- **Architecture**: `784 -> 128 -> ReLU -> 64 -> ReLU -> 10`
- **Optimizer**: Adam ($\text{lr} = 0.005$, StepLR $\gamma=0.5$)
- **Recorded test accuracy**: **94.80%** (5,000 training samples; 1,000 test samples)

### CNN
- **Architecture**: `Conv2d(1->8) -> ReLU -> MaxPool2d -> Conv2d(8->16) -> ReLU -> MaxPool2d -> Flatten -> Linear`
- **Optimizer**: Adam ($\text{lr} = 0.005$)
- **Recorded test accuracy**: **95.80%** (2,000 training samples; 500 test samples)

---

## PyTorch Comparisons

`tests/test_pytorch_compat.py` compares forward values and gradients for these selected operations using identical inputs and weights:

| Operation | Forward tolerance | Gradient tolerance |
| :--- | :--- | :--- |
| `Linear + ReLU` | `1e-5` | `1e-5` |
| `CrossEntropyLoss` | `1e-5` | `1e-5` |
| `Conv2d` (im2col) | `1e-4` | `1e-4` |

---

## Installation and Testing

```bash
# Install in editable mode with development dependencies
pip install -e .[dev]

# Run the test suite
python -m pytest tests -v

# Run MNIST MLP training
python examples/train_mnist_mlp.py

# Run MNIST CNN training
python examples/train_mnist_cnn.py

# Run Conv2d vectorization profiler
python examples/profile_conv.py
```

---

## Limitations

1. **In-place Operations**: TinyTorch does not currently support in-place tensor mutations (e.g. `x += y` during forward pass) to preserve autograd graph history safety.
2. **CPU NumPy Execution**: Computation runs on CPU via NumPy arrays. Adding a GPU backend (via CuPy or WebGPU/OpenCL) would enable massive throughput scaling.
3. **Higher-Order Gradients**: Current backward pass returns numpy arrays for gradients rather than wrapped `Tensor` graph nodes (second-order derivatives like $H \cdot v$ Hessian-vector products require wrapping backward ops into graph nodes).

---

## 📄 License

MIT License
