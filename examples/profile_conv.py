import time
import numpy as np
from tinytorch.tensor import Tensor
import tinytorch.nn as nn
import tinytorch.optim as optim
import tinytorch.ops as ops


def profile_training_loop():
    print("==================================================")
    print("   TinyTorch Conv2d Vectorization Profiler       ")
    print("==================================================")

    # Synthetic batch of images: 32 images, 1 channel, 14x14
    N, C, H, W = 16, 1, 14, 14
    X_data = np.random.randn(N, C, H, W).astype(np.float32)
    y_data = np.random.randint(0, 10, size=N)

    # 1. Unvectorized Conv2d Layer setup
    conv_unvec = nn.Conv2d(1, 8, kernel_size=3, padding=1, vectorized=False)
    linear = nn.Linear(8 * 14 * 14, 10)
    optimizer_unvec = optim.SGD(conv_unvec.parameters() + linear.parameters(), lr=0.01)

    print("\n---> Profiling UNVECTORIZED (Python loops) Training Step...")
    start_unvec = time.perf_counter()
    num_runs = 3
    for _ in range(num_runs):
        optimizer_unvec.zero_grad()
        x = Tensor(X_data)
        h = ops.relu(conv_unvec(x))
        h_flat = ops.flatten(h)
        logits = linear(h_flat)
        loss = ops.cross_entropy(logits, y_data)
        loss.backward()
        optimizer_unvec.step()
    end_unvec = time.perf_counter()
    time_unvectorized = (end_unvec - start_unvec) / num_runs

    # 2. Vectorized Conv2d Layer setup
    conv_vec = nn.Conv2d(1, 8, kernel_size=3, padding=1, vectorized=True)
    # Copy weights for exact match
    conv_vec.weight.data = conv_unvec.weight.data.copy()
    conv_vec.bias.data = conv_unvec.bias.data.copy()
    optimizer_vec = optim.SGD(conv_vec.parameters() + linear.parameters(), lr=0.01)

    print("\n---> Profiling VECTORIZED (NumPy array ops) Training Step...")
    start_vec = time.perf_counter()
    for _ in range(num_runs):
        optimizer_vec.zero_grad()
        x = Tensor(X_data)
        h = ops.relu(conv_vec(x))
        h_flat = ops.flatten(h)
        logits = linear(h_flat)
        loss = ops.cross_entropy(logits, y_data)
        loss.backward()
        optimizer_vec.step()
    end_vec = time.perf_counter()
    time_vectorized = (end_vec - start_vec) / num_runs

    speedup = time_unvectorized / time_vectorized if time_vectorized > 0 else 0.0

    print("\n==================================================")
    print("           PROFILING RESULTS SUMMARY              ")
    print("==================================================")
    print(f"Unvectorized (Python loops) : {time_unvectorized*1000:.2f} ms / step")
    print(f"Vectorized (NumPy im2col)   : {time_vectorized*1000:.2f} ms / step")
    print(f"Speedup Factor              : {speedup:.2f}x faster")
    print("==================================================\n")

    return time_unvectorized, time_vectorized, speedup


if __name__ == "__main__":
    profile_training_loop()
