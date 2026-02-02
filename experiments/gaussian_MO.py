from src import load_datasets, GMBarron, train_mse, estimate_GM_model
import torch
import numpy as np
import matplotlib.pyplot as plt

# def train_and_validate_gaussian(d:int) -> torch.float:



d = 3
filename = f"data/gaussian_{d}.npz"
train_loader, test_loader = load_datasets(filename)

# Verify train_loader
print("batches:", len(train_loader))
print("samples:", len(train_loader.dataset))
print("batch size:", train_loader.batch_size)
x, y = next(iter(train_loader))
print("x:", x.shape, x.dtype, x.device)
print("y:", y.shape, y.dtype, y.device)

print("\n")

# Verify test_loader
print("batches:", len(test_loader))
print("samples:", len(test_loader.dataset))
print("batch size:", test_loader.batch_size)
x, y = next(iter(test_loader))
print("x:", x.shape, x.dtype, x.device)
print("y:", y.shape, y.dtype, y.device)


# Train model
d_in = d
d_out = 1
K = 1000
S = 500
model = GMBarron(d_in, K, S, d_out,"tanh")
opt = torch.optim.Adam(model.parameters(),lr=1e-3)
epochs = 1000

train_mse(model, train_loader, opt, epochs)

batch = next(iter(test_loader))
x, y = batch
pred_m, pred_v = estimate_GM_model(model, x, 2000)

y_vis = y.detach().numpy().reshape(-1)
ypred_vis = pred_m.detach().numpy().reshape(-1)

plt.scatter(y_vis, ypred_vis, alpha=0.1)
plt.plot(y_vis, y_vis, '--')
plt.xlabel("y (true)")
plt.ylabel("y_pred")
plt.axis("equal")
plt.savefig("experiments/MO.png")

print(pred_m)