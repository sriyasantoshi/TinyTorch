from __future__ import annotations
import os
import numpy as np
from typing import Tuple


def load_mnist(num_train: int = 5000, num_test: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load or generate MNIST dataset.
    
    Returns:
        X_train: (num_train, 784) float32 normalized to [0, 1]
        y_train: (num_train,) int64
        X_test:  (num_test, 784) float32 normalized to [0, 1]
        y_test:  (num_test,) int64
    """
    try:
        from sklearn.datasets import fetch_openml
        print("Fetching MNIST dataset via OpenML...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(np.int64)
        
        X_train, y_train = X[:num_train], y[:num_train]
        X_test, y_test = X[60000 : 60000 + num_test], y[60000 : 60000 + num_test]
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"Could not load full MNIST via OpenML ({e}). Falling back to load_digits dataset...")
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0
        y = digits.target.astype(np.int64)
        
        # Upsample digits from 8x8 to 28x28 for MNIST shape (784 features)
        N = X.shape[0]
        X_28 = np.zeros((N, 784), dtype=np.float32)
        X_reshaped = X.reshape(N, 8, 8)
        for i in range(N):
            # Repeat elements to form 28x28-like grid
            img = np.repeat(np.repeat(X_reshaped[i], 3, axis=0), 3, axis=1) # 24x24
            X_28[i, :24*24] = img.reshape(-1)
            
        n_train = min(num_train, int(0.8 * N))
        n_test = min(num_test, N - n_train)
        return X_28[:n_train], y[:n_train], X_28[n_train:n_train+n_test], y[n_train:n_train+n_test]
