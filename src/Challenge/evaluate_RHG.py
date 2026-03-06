from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms

from model import ConvolutionalNet
from model import SimpleResNet

try:
    from .dataset import SignosDataset
    from .model import ConvolutionalNeuralNetwork, MultiTaskVGGDropout, VGG, MultiTaskVGG
    from .train import BATCH_SIZE, IMAGE_SIZE
except ImportError:
    from dataset import SignosDataset
    from model import ConvolutionalNeuralNetwork, MultiTaskVGGDropout, VGG, MultiTaskVGG
    from train import BATCH_SIZE, IMAGE_SIZE


def evaluate_and_plot(loader, model, dataset_name, output_folder, class_names, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)

            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Confusion matrix
    label_ids = np.arange(len(class_names))
    cm = confusion_matrix(all_targets, all_predictions, labels=label_ids)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicción")
    plt.ylabel("Realidad (Target)")
    plt.title(f"Matriz de Confusión ({dataset_name})")
    plt.savefig(output_folder / f"confusion_matrix_{dataset_name}2.png")
    plt.show()

    # Metrics
    accuracy = float(np.mean(all_predictions == all_targets))
    print(f"\n[{dataset_name}] Accuracy: {accuracy:.4f}")

    prec, rec, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, labels=label_ids, average=None, zero_division=0
    )

    print(f"{'CLASE':<10} {'PRECISION':<10} {'RECALL':<10} {'F1-SCORE':<10} {'CANTIDAD (Support)'}")
    print("-" * 60)
    for i in range(len(class_names)):
        print(f"{class_names[i]:<10} {prec[i]:.2f}       {rec[i]:.2f}       {f1[i]:.2f}       {support[i]}")

    metrics = {
        "Accuracy": accuracy,
        "Clase": class_names,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Cantidad": support,
    }
    return metrics


def save_metrics_as_picture(metrics, filepath):
    df = pd.DataFrame(metrics).round(6)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)


def resolve_checkpoint_path(base_outs: Path) -> Path:
    """Resolve best_model checkpoint path trying common project locations."""
    candidates = [
        base_outs / "Challenge" / "best_model.pth",
        base_outs / "best_model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No se encontró el checkpoint best_model.pth. Rutas intentadas:\n"
        f"{tried}\n"
        "Primero ejecuta train.py para generar el modelo."
    )


if __name__ == "__main__":
    base_outs = Path(__file__).parent.parent.parent / "outs" / "Challenge"
    output_folder = base_outs / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # Transform (sin augment)
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ]
    )

    train_subset = SignosDataset(Mode="train", transform=transform)
    val_subset = SignosDataset(Mode="val", transform=transform)
    test_dataset = SignosDataset(Mode="test", transform=transform)

    # OJO: antes tenías train_subset.data.classes; aquí usamos el atributo estándar
    # Si tu dataset lo tiene como train_subset.classes, perfecto.
    # Si lo tiene como train_subset.data.classes, cambia la siguiente línea.
    class_names = getattr(train_subset, "classes", None)
    if class_names is None:
        class_names = train_subset.data.classes

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load checkpoint
    checkpoint_path = resolve_checkpoint_path(base_outs)
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Build correct model depending on checkpoint format
    if "fc8.weight" in checkpoint:
        checkpoint_output_dim = checkpoint["fc8.weight"].shape[0]
        model = VGG(output_dim=checkpoint_output_dim)

    elif "classifier.4.weight" in checkpoint:
        # MultiTaskVGGDropout (con Dropout en classifier)
        checkpoint_output_dim = checkpoint["classifier.4.weight"].shape[0]
        model = MultiTaskVGGDropout(output_dim=checkpoint_output_dim)

    elif "classifier.3.weight" in checkpoint:
        # MultiTaskVGG (sin Dropout dentro del classifier)
        checkpoint_output_dim = checkpoint["classifier.3.weight"].shape[0]
        model = MultiTaskVGG(output_dim=checkpoint_output_dim)

        # Si quisieras forzar cargar en Dropout, usa este remapeo en su lugar:
        # checkpoint_output_dim = checkpoint["classifier.3.weight"].shape[0]
        # checkpoint["classifier.4.weight"] = checkpoint.pop("classifier.3.weight")
        # checkpoint["classifier.4.bias"] = checkpoint.pop("classifier.3.bias")
        # model = MultiTaskVGGDropout(output_dim=checkpoint_output_dim)

    else:
        raise KeyError(
            "Unsupported checkpoint format: expected keys 'fc8.weight' or "
            "'classifier.4.weight' (Dropout) or 'classifier.3.weight' (no Dropout)."
        )

    # Ajustar class_names si output_dim no coincide
    if checkpoint_output_dim != len(class_names):
        print(
            "Warning: checkpoint output_dim "
            f"({checkpoint_output_dim}) != dataset classes ({len(class_names)})."
        )
        if checkpoint_output_dim > len(class_names):
            extra = [f"extra_class_{i}" for i in range(len(class_names), checkpoint_output_dim)]
            class_names = list(class_names) + extra
        else:
            class_names = list(class_names)[:checkpoint_output_dim]

    # Load weights and move to device
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    metrics = {}

    metrics["train"] = evaluate_and_plot(train_loader, model, "train", output_folder, class_names, device)
    metrics["validation"] = evaluate_and_plot(val_loader, model, "validation", output_folder, class_names, device)
    metrics["test"] = evaluate_and_plot(test_loader, model, "test", output_folder, class_names, device)

    # Save metrics as CSVs
    pd.DataFrame(metrics["train"]).to_csv(output_folder / "metrics_train2.csv", index=False)
    pd.DataFrame(metrics["validation"]).to_csv(output_folder / "metrics_validation2.csv", index=False)
    pd.DataFrame(metrics["test"]).to_csv(output_folder / "metrics_test2.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_folder / "metrics2.csv")

    # Save metrics as images
    save_metrics_as_picture(metrics["train"], output_folder / "metrics_train2.png")
    save_metrics_as_picture(metrics["validation"], output_folder / "metrics_validation2.png")
    save_metrics_as_picture(metrics["test"], output_folder / "metrics_test2.png")

    print("Evaluation complete!")