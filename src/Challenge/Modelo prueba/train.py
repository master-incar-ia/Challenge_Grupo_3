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
from tqdm import tqdm

try:
    from .dataset import SignosDataset, build_augment_transform, build_eval_transform
    from .model import Segmentation, VGG
except ImportError:
    from dataset import SignosDataset, build_augment_transform, build_eval_transform
    from model import Segmentation, VGG


BATCH_SIZE = 512
IMAGE_SIZE = (32, 32)
NUM_EPOCHS = 100
CACHE_IN_RAM = True
PIPELINE_TO_RUN = "vgg"  # "segmentation" | "vgg"


def get_device(force: str = "auto") -> torch.device:
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def profile_pipeline(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_batches: int = 30,
) -> None:
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
        if isinstance(outputs, tuple):
            outputs = outputs[0]
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


def build_pseudo_masks(inputs: torch.Tensor) -> torch.Tensor:
    gray = inputs.mean(dim=1, keepdim=True)
    threshold = gray.mean(dim=(2, 3), keepdim=True)
    return (gray > threshold).float()


def build_cached_tensor_dataset(dataset, name: str) -> TensorDataset:
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


def build_loaders(device: torch.device):
    eval_transform = build_eval_transform(image_size=IMAGE_SIZE)
    augment_transform = build_augment_transform(image_size=IMAGE_SIZE)

    train_subset = SignosDataset(
        Mode="train",
        transform=eval_transform,
        augmented_transform=augment_transform,
        include_original=True,
    )
    val_subset = SignosDataset(Mode="val", transform=eval_transform)
    class_count = len(train_subset.classes)

    if CACHE_IN_RAM:
        train_subset = build_cached_tensor_dataset(train_subset, name="train")
        val_subset = build_cached_tensor_dataset(val_subset, name="val")

    pin_memory = device.type == "cuda"
    num_workers = 0 if CACHE_IN_RAM else min(8, os.cpu_count() or 1)
    persistent_workers = num_workers > 0
    prefetch_factor = 4 if num_workers > 0 else None

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, class_count


def train_segmentation_model(output_folder: Path, device: torch.device):
    train_loader, val_loader, _ = build_loaders(device)

    model = Segmentation().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        profile_pipeline(train_loader, model, nn.MSELoss(), optimizer, device)

    best_val_loss = float("inf")
    best_model_path = output_folder / "best_segmentation.pth"
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(NUM_EPOCHS), desc="Train segmentation"):
        model.train()
        train_loss = 0.0

        for inputs, _ in train_loader:
            inputs_cuda = inputs.to(device, non_blocking=True)
            mask_targets = build_pseudo_masks(inputs_cuda)

            with autocast("cuda", enabled=device.type == "cuda"):
                mask_logits = model(inputs_cuda)
                loss = criterion(mask_logits, mask_targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs_cuda = inputs.to(device, non_blocking=True)
                mask_targets = build_pseudo_masks(inputs_cuda)
                with autocast("cuda", enabled=device.type == "cuda"):
                    mask_logits = model(inputs_cuda)
                    loss = criterion(mask_logits, mask_targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    plt.figure(figsize=(10, 5))
    plt.plot(range(NUM_EPOCHS), train_losses, label="Train Loss")
    plt.plot(range(NUM_EPOCHS), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Segmentation Training Loss")
    plt.savefig(output_folder / "loss_plot_segmentation.png")

    print(f"Best segmentation val loss: {best_val_loss:.4f}, saved at {best_model_path}")


def train_vgg_model(output_folder: Path, device: torch.device):
    train_loader, val_loader, class_count = build_loaders(device)

    segmentation_checkpoint = output_folder / "best_segmentation.pth"
    if not segmentation_checkpoint.exists():
        raise FileNotFoundError(
            f"No se encontró {segmentation_checkpoint}. Entrena primero segmentation."
        )

    segmentation_model = Segmentation().to(device)
    segmentation_model.load_state_dict(torch.load(segmentation_checkpoint, map_location=device))
    segmentation_model.eval()
    for parameter in segmentation_model.parameters():
        parameter.requires_grad = False

    model = VGG(output_dim=class_count, in_channels=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        profile_pipeline(train_loader, model, criterion, optimizer, device)

    best_val_loss = float("inf")
    best_model_path = output_folder / "best_vgg.pth"
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(NUM_EPOCHS), desc="Train VGG"):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs_cuda = inputs.to(device, non_blocking=True)
            targets_cuda = targets.to(device, non_blocking=True)

            with torch.no_grad():
                mask_logits = segmentation_model(inputs_cuda)
                mask_prob = torch.sigmoid(mask_logits)

            classifier_input = torch.cat([inputs_cuda, mask_prob], dim=1)

            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(classifier_input)
                loss = criterion(logits, targets_cuda)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs_cuda = inputs.to(device, non_blocking=True)
                targets_cuda = targets.to(device, non_blocking=True)
                mask_logits = segmentation_model(inputs_cuda)
                mask_prob = torch.sigmoid(mask_logits)
                classifier_input = torch.cat([inputs_cuda, mask_prob], dim=1)

                with autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(classifier_input)
                    loss = criterion(logits, targets_cuda)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    plt.figure(figsize=(10, 5))
    plt.plot(range(NUM_EPOCHS), train_losses, label="Train Loss")
    plt.plot(range(NUM_EPOCHS), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("VGG (RGB+Mask) Training Loss")
    plt.savefig(output_folder / "loss_plot_vgg.png")

    print(f"Best VGG val loss: {best_val_loss:.4f}, saved at {best_model_path}")


if __name__ == "__main__":
    torch.manual_seed(42)

    if sys.gettrace() is not None:
        print("WARNING: Running under debugger; performance can be much slower. Run from terminal.")

    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")
    print(f"Using device: {device}")
    print(f"Pipeline: {PIPELINE_TO_RUN}")

    if PIPELINE_TO_RUN == "segmentation":
        train_segmentation_model(output_folder, device)
    elif PIPELINE_TO_RUN == "vgg":
        train_vgg_model(output_folder, device)
    else:
        raise ValueError("PIPELINE_TO_RUN must be 'segmentation' or 'vgg'.")
