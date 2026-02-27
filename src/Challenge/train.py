from pathlib import Path
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from dataset import SignosDataset 
from model import VGG


BATCH_SIZE = 64
IMAGE_SIZE = (32, 32)


def get_device(force: str = "auto") -> torch.device:
    """Return a torch.device based on the `force` option.

    force: 'auto'|'cpu'|'cuda' - when 'auto' will pick cuda if available.
    """
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def profile_pipeline(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_batches: int = 100,
) -> None:
    """Profile data loading, H2D transfer and compute time on a few batches."""
    model.train()

    batches_to_profile = min(num_batches, len(train_loader))
    if batches_to_profile == 0:
        print("[PROFILE] No batches available to profile.")
        return

    data_time = 0.0
    transfer_time = 0.0
    compute_time = 0.0
    total_samples = 0

    loader_iter = iter(train_loader)

    for _ in range(batches_to_profile):
        fetch_start = time.perf_counter()
        inputs, targets = next(loader_iter)
        fetch_end = time.perf_counter()
        data_time += fetch_end - fetch_start

        transfer_start = time.perf_counter()
        inputs_cuda = inputs.to(device, non_blocking=True)
        targets_cuda = targets.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        transfer_end = time.perf_counter()
        transfer_time += transfer_end - transfer_start

        compute_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs_cuda)
        loss = criterion(outputs, targets_cuda)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_end = time.perf_counter()
        compute_time += compute_end - compute_start

        total_samples += inputs.shape[0]

    optimizer.zero_grad(set_to_none=True)

    total_time = data_time + transfer_time + compute_time
    avg_data_ms = (data_time / batches_to_profile) * 1000
    avg_transfer_ms = (transfer_time / batches_to_profile) * 1000
    avg_compute_ms = (compute_time / batches_to_profile) * 1000
    samples_per_sec = total_samples / total_time if total_time > 0 else 0.0

    print("\n[PROFILE] ---- Training pipeline breakdown ----")
    print(f"[PROFILE] Batches: {batches_to_profile}")
    print(f"[PROFILE] Data load: {avg_data_ms:.2f} ms/batch ({100 * data_time / total_time:.1f}%)")
    print(
        f"[PROFILE] H2D copy : {avg_transfer_ms:.2f} ms/batch "
        f"({100 * transfer_time / total_time:.1f}%)"
    )
    print(f"[PROFILE] Compute  : {avg_compute_ms:.2f} ms/batch ({100 * compute_time / total_time:.1f}%)")
    print(f"[PROFILE] Throughput: {samples_per_sec:.1f} samples/s")
    print("[PROFILE] -------------------------------------\n")


def build_cached_tensor_dataset(dataset, name: str) -> TensorDataset:
    """Preload transformed samples into RAM to avoid disk/PIL bottlenecks at runtime."""
    print(f"Caching {name} dataset in RAM...")
    images = []
    labels = []
    for image, label in tqdm(dataset, desc=f"Caching {name}", leave=False):
        images.append(image)
        labels.append(label)

    image_tensor = torch.stack(images)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    print(
        f"{name} cached: {len(dataset)} samples "
        f"({image_tensor.numel() * image_tensor.element_size() / (1024 ** 3):.2f} GB images)"
    )
    return TensorDataset(image_tensor, label_tensor)


def train_model(output_folder: Path, device: torch.device):
    # Data augmentation
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ]
    )

    # Cargas el bloque de 50k de CIFAR10, y luego lo divides en dos (90% entrenamiento, 10% validación interna)
    train_subset = SignosDataset(Mode="train", transform=transform)

    # Divides ese bloque en dos (90% entrenamiento, 10% validación interna)
    val_subset=SignosDataset(Mode="val", transform=transform)

    cache_in_ram = True
    if cache_in_ram:
        train_subset = build_cached_tensor_dataset(train_subset, name="train")
        val_subset = build_cached_tensor_dataset(val_subset, name="val")

    # Create DataLoaders for the datasets
    pin_memory = device.type == "cuda"
    num_workers = 0 if cache_in_ram else min(8, os.cpu_count() or 1)
    persistent_workers = num_workers > 0
    prefetch_factor = 4 if num_workers > 0 else None

    train_loader = DataLoader(
        train_subset,
        batch_size=512,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=512,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # Define the model, loss function, and optimizer
    output_dim = len(train_subset.data.classes)
    model = VGG(output_dim=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(
            f"DataLoader workers={num_workers}, pin_memory={pin_memory}, "
            f"persistent_workers={persistent_workers}"
        )

    if device.type == "cuda":
        print("Running quick pipeline profile (30 batches)...")
        profile_pipeline(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_batches=30,
        )

    # Training loop with validation and saving best weights
    num_epochs = 80
    best_val_loss = float("inf")
    best_model_path = output_folder / "best_model.pth"

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            # Forward pass
            inputs_cuda = inputs.to(device, non_blocking=True)
            targets_cuda = targets.to(device, non_blocking=True)

            with autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(inputs_cuda)
                loss = criterion(outputs, targets_cuda)

            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs_cuda = inputs.to(device, non_blocking=True)
                targets_cuda = targets.to(device, non_blocking=True)
                with autocast("cuda", enabled=device.type == "cuda"):
                    outputs = model(inputs_cuda)
                    loss = criterion(outputs, targets_cuda)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

    print(f"Best validation loss: {best_val_loss:.4f}, Model saved to {best_model_path}")

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Save the plot to the outs/ folder
    plt.savefig(output_folder / "loss_plot.png")


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    if sys.gettrace() is not None:
        print("WARNING: Running under debugger; performance can be much slower. Run from terminal for valid benchmarks.")

    # Create output folder based on file folder
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")  # choices are "auto", "cpu", "cuda"
    print(f"Using device: {device}")
    train_model(output_folder, device=device)
