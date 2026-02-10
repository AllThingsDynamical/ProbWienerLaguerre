import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_datasets(
    filename: str,
    batch_size: int = 128,
):
    """
    Reads X_train, Y_train, X_test, Y_test from an .npz file,
    transposes them, and returns PyTorch DataLoaders.
    """
    data = np.load(filename)

    Xtr = data["X_train"].T
    Ytr = data["Y_train"].T
    Xte = data["X_test"].T
    Yte = data["Y_test"].T

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Ytr = torch.tensor(Ytr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    Yte = torch.tensor(Yte, dtype=torch.float32)

    train_ds = TensorDataset(Xtr, Ytr)
    test_ds  = TensorDataset(Xte, Yte)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_mse(
    model,
    train_loader,
    optimizer,
    epochs: int = 100,
    device: str | torch.device = "cpu",
):
    """
    Simple training loop with MSE loss.
    """
    model.to(device)
    model.train()

    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        if epoch % 10 == 1:
            print(f"epoch {epoch:03d} | mse {avg_loss:.6e}")


def evaluate_mse(
    model,
    test_loader,
    device: str | torch.device = "cpu",
):
    """
    Evaluate a trained model using MSE.
    Returns average MSE over the test set.
    """
    model.to(device)
    model.eval()

    criterion = torch.nn.MSELoss(reduction="sum")
    total_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()

    mse = total_loss / len(test_loader.dataset)
    return mse
