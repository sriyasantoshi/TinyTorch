import time
import numpy as np
import matplotlib.pyplot as plt
from tinytorch.tensor import Tensor
import tinytorch.nn as nn
import tinytorch.optim as optim
from tinytorch.utils.dataset import load_mnist


def train_mlp():
    print("=== Training TinyTorch MLP on MNIST ===")
    X_train, y_train, X_test, y_test = load_mnist(num_train=5000, num_test=1000)

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.StepLR(optimizer, step_size=5, gamma=0.5)

    batch_size = 64
    epochs = 10
    num_samples = len(X_train)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        permutation = np.random.permutation(num_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        epoch_loss = 0.0
        num_batches = 0

        start_time = time.perf_counter()
        for i in range(0, num_samples, batch_size):
            x_batch = Tensor(X_train_shuffled[i : i + batch_size])
            y_batch = y_train_shuffled[i : i + batch_size]

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data)
            num_batches += 1

        scheduler.step()
        epoch_time = time.perf_counter() - start_time
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Evaluate on test set
        model.eval()
        x_test_tensor = Tensor(X_test)
        logits_test = model(x_test_tensor)
        preds = np.argmax(logits_test.data, axis=1)
        acc = float(np.mean(preds == y_test)) * 100.0
        test_accuracies.append(acc)

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | Time: {epoch_time:.3f}s | "
            f"Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}% | LR: {optimizer.lr:.6f}"
        )

    # Save training curve plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, "b-o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), test_accuracies, "g-o", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("MLP Test Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("mnist_mlp_training_curve.png", dpi=150)
    plt.close()
    print("Saved training curve to mnist_mlp_training_curve.png")
    return train_losses, test_accuracies


if __name__ == "__main__":
    train_mlp()
