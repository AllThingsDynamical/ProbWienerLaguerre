from src import load_datasets, GMBarron, train_mse, estimate_GM_model
import torch
import numpy as np
import matplotlib.pyplot as plt

# def train_and_validate_gaussian(d:int) -> torch.float:



d = 2
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
K = 100
S = 1000
model = GMBarron(d_in, K, S, d_out,"gelu")
opt = torch.optim.Adam(model.parameters(),lr=1e-3)
epochs = 200

train_mse(model, train_loader, opt, epochs)

batch = next(iter(test_loader))
x, y = batch
pred_m, pred_v = estimate_GM_model(model, x, 2000)

x_plot = x[:,0]
y_plot = x[:,1]

c_plot = y
v_plot = pred_m


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

sc1 = axes[0].scatter(x_plot, y_plot, c=c_plot, cmap="viridis")
axes[0].set_title("Actual")
plt.colorbar(sc1, ax=axes[0])

sc2 = axes[1].scatter(x_plot, y_plot, c=v_plot, cmap="viridis")
axes[1].set_title("Estimated")
plt.colorbar(sc2, ax=axes[1])

plt.tight_layout()
plt.show()