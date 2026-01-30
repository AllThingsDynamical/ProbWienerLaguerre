from src import load_datasets, AtomicBarron, train_mse, evaluate_mse
import torch
import numpy as np

def train_and_validate_gaussian(d:int) -> torch.float:
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
    model = AtomicBarron(d_in, K, d_out,"gelu")
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)
    epochs = 200

    train_mse(model, train_loader, opt, epochs)

    # Validate model
    mse = evaluate_mse(model, test_loader)
    print(f"test mse = {mse:.6e}")
    return mse


nruns = 5 
validation_runs = []
for run in range(nruns):
    validation_errors = []
    dimensions = [2, 3, 5, 10, 20]
    for d in dimensions:
        err = train_and_validate_gaussian(d)
        print("\n\n")
        print(f"Dimension : {d} and Validation Error: {err}")
        validation_errors.append(np.array(err))
    validation_runs.append(validation_errors)

print(np.mean(validation_runs, axis=0))