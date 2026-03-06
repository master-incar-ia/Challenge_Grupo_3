from pathlib import Path
import os
import sys
import time
import copy
from typing import Set, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import torchvision.transforms as T

from dataset import SignosDataset, build_augment_transform, build_eval_transform
from model import MultiTaskVGG
# from model import ConvolutionalNet
# from model import SimpleResNet


BATCH_SIZE = 512
IMAGE_SIZE = (32, 32)
SEGMENTATION_WEIGHT = 1.0  


def get_device(force: str = "auto") -> torch.device:
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_to_tensor(transform):
    """Elimina ToTensor() de un Compose para poder aplicarlo sobre tensores ya cacheados."""
    if isinstance(transform, T.Compose):
        filtered = [t for t in transform.transforms if not isinstance(t, T.ToTensor)]
        return T.Compose(filtered)
    return transform


def profile_pipeline(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_batches: int = 100,
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
        inputs_dev = inputs.to(device, non_blocking=True)
        targets_dev = targets.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        transfer_end = time.perf_counter()
        transfer_time += transfer_end - transfer_start

        compute_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs_dev)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, targets_dev)
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
    print(f"[PROFILE] H2D copy : {avg_transfer_ms:.2f} ms/batch ({100 * transfer_time / total_time:.1f}%)")
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


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.num_bad = 0
            return False
        self.num_bad += 1
        return self.num_bad >= self.patience


class AugmentOnTensorDataset(torch.utils.data.Dataset):
    """Aplica augment sobre un TensorDataset cacheado (entrada ya es torch.Tensor)."""
    def __init__(self, base, transform=None):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class ClassConditionalAugmentDataset(torch.utils.data.Dataset):
    """Aplica transform extra SOLO si y está en target_labels."""
    def __init__(self, base_dataset, target_labels: Set[int], extra_transform, p: float = 0.3):
        self.base = base_dataset
        self.target_labels = set(target_labels)
        self.extra_transform = extra_transform
        self.p = float(p)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if (y in self.target_labels) and (torch.rand(()) < self.p):
            x = self.extra_transform(x)
        return x, y


def train_model(output_folder: Path, device: torch.device):
    # Transforms base
    eval_transform = build_eval_transform(image_size=IMAGE_SIZE)
    augment_transform = build_augment_transform(image_size=IMAGE_SIZE)

    # Si vamos a cachear tensores, quitamos ToTensor() del augment
    augment_transform = remove_to_tensor(augment_transform)

    # Dataset base (SIN augmented_transform dentro) -> así cacheamos un “base limpio”
    train_base = SignosDataset(
        Mode="train",
        transform=eval_transform,
        augmented_transform=None,  
        include_original=True,
    )
    val_base = SignosDataset(Mode="val", transform=eval_transform)

    class_count = len(train_base.classes)
    output_dim = class_count

    # Cache en RAM
    train_cached = build_cached_tensor_dataset(train_base, name="train")
    val_cached = build_cached_tensor_dataset(val_base, name="val")

    # Augment dinámico encima del cache (rápido)
    train_dataset = AugmentOnTensorDataset(train_cached, transform=augment_transform)

    # Clases difíciles
    hard_letters = {"M", "N", "T", "F", "Q", "S", "D", "C"}
    letter_to_idx = {c: i for i, c in enumerate(train_base.classes)}
    hard_class_ids = {letter_to_idx[ch] for ch in hard_letters if ch in letter_to_idx}

    print(f"Classes: {train_base.classes}")
    print(f"Hard letters: {sorted(list(hard_letters))}")
    print(f"Hard class ids: {sorted(list(hard_class_ids))}")

    # Hard augment “barato” (tensor-safe; no ToTensor aquí)
    hard_aug = T.Compose(
        [
            T.RandomApply([T.RandomRotation(degrees=15)], p=0.5),

        ]
    )
    hard_aug = remove_to_tensor(hard_aug)

    if len(hard_class_ids) > 0:
        train_dataset = ClassConditionalAugmentDataset(
            base_dataset=train_dataset,
            target_labels=hard_class_ids,
            extra_transform=hard_aug,
            p=0.3,
        )

    # DataLoaders
    pin_memory = device.type == "cuda"
    num_workers = 0  # cacheado -> 0 suele ir muy bien
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_cached,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0,
    )

    # Modelo
    model = MultiTaskVGG(output_dim=output_dim).to(device)
    # model = ConvolutionalNet(output_dim=output_dim).to(device)
    # model = SimpleResNet(output_dim=output_dim).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-6,
    )

    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print("Running quick pipeline profile (30 batches)...")
        profile_pipeline(
            train_loader=train_loader,
            model=model,
            criterion=criterion_cls,
            optimizer=optimizer,
            device=device,
            num_batches=30,
        )

    # Training loop
    num_epochs = 150
    best_val_loss = float("inf")
    best_model_path = output_folder / "best_model.pth"
    early_stopper = EarlyStopping(patience=15, min_delta=1e-4)
    best_state = None

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs_dev = inputs.to(device, non_blocking=True)
            targets_dev = targets.to(device, non_blocking=True)
            mask_targets = build_pseudo_masks(inputs_dev)

            with autocast("cuda", enabled=device.type == "cuda"):
                class_logits, mask_logits = model(inputs_dev)
                loss_cls = criterion_cls(class_logits, targets_dev)
                loss_seg = criterion_seg(mask_logits, mask_targets)
                loss = loss_cls + SEGMENTATION_WEIGHT * loss_seg

            train_loss += float(loss.item())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss /= max(1, len(train_loader))
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs_dev = inputs.to(device, non_blocking=True)
                targets_dev = targets.to(device, non_blocking=True)
                mask_targets = build_pseudo_masks(inputs_dev)

                with autocast("cuda", enabled=device.type == "cuda"):
                    class_logits, mask_logits = model(inputs_dev)
                    loss_cls = criterion_cls(class_logits, targets_dev)
                    loss_seg = criterion_seg(mask_logits, mask_targets)
                    loss = loss_cls + SEGMENTATION_WEIGHT * loss_seg

                val_loss += float(loss.item())

        val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_model_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}"
            )

        if early_stopper.step(val_loss):
            print(f"Early stopping at epoch {epoch + 1}. Best val loss: {best_val_loss:.4f}")
            break

    print(f"Best validation loss: {best_val_loss:.4f}, Model saved to {best_model_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(output_folder / "loss_plot.png")


if __name__ == "__main__":
    torch.manual_seed(42)

    if sys.gettrace() is not None:
        print("WARNING: Running under debugger; performance can be much slower. Run from terminal for valid benchmarks.")

    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")
    print(f"Using device: {device}")

    train_model(output_folder, device=device)