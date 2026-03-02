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
    from .model import ConvolutionalNeuralNetwork, MultiTaskVGG, VGG
    from .train import BATCH_SIZE, IMAGE_SIZE
except ImportError:
    from dataset import SignosDataset
    from model import ConvolutionalNeuralNetwork, MultiTaskVGG, VGG
    from train import BATCH_SIZE, IMAGE_SIZE


def evaluate_and_plot(loader, model, dataset_name, output_folder, class_names):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Set the confusion matrix
    label_ids = np.arange(len(class_names))
    map = confusion_matrix(all_targets, all_predictions, labels=label_ids)

    # Plot and save the confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        map,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicción")
    plt.ylabel("Realidad (Target)")
    plt.title("Matriz de Confusión")
    plt.savefig(output_folder / f"confusion_matrix_{dataset_name}.png")
    plt.show()

    # Let's obtain the accuracy, precision, recall and F1 score for the dataset
    accuracy = np.mean(all_predictions == all_targets)
    print(f"Accuracy: {accuracy:.4f}")

    prec, rec, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, labels=label_ids, average=None, zero_division=0
    )

    print(f"{'CLASE':<10} {'PRECISION':<10} {'RECALL':<10} {'F1-SCORE':<10} {'CANTIDAD (Support)'}")
    print("-" * 60)

    for i in range(len(class_names)):
        print(
            f"{class_names[i]:<10} {prec[i]:.2f}       {rec[i]:.2f}       {f1[i]:.2f}       {support[i]}"
        )

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
    # Create a DataFrame
    df = pd.DataFrame(metrics)

    # Round the values to 6 decimal places
    df = df.round(6)

    # Plot the table and save as an image
    fig, ax = plt.subplots(figsize=(16, 10))  # set size frame
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc="center", loc="center"
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
    # Set the seed for reproducibility
    torch.manual_seed(42)
    # Data augmentation

    MEAN = [0.50956494, 0.50055039, 0.49491626] 
    STD = [0.07214967, 0.09587376, 0.11140345]
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    train_subset = SignosDataset(Mode="train", transform=transform)
    val_subset = SignosDataset(Mode="val", transform=transform)
    test_dataset = SignosDataset(Mode="test", transform=transform)
    class_names = train_subset.data.classes

    # Create DataLoaders for the datasets
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the best model weights
    checkpoint_path = resolve_checkpoint_path(base_outs)
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "fc8.weight" in checkpoint:
        checkpoint_output_dim = checkpoint["fc8.weight"].shape[0]
        model = VGG(output_dim=checkpoint_output_dim)
    elif "classifier.3.weight" in checkpoint:
        checkpoint_output_dim = checkpoint["classifier.3.weight"].shape[0]
        model = MultiTaskVGG(output_dim=checkpoint_output_dim)
    else:
        raise KeyError(
            "Unsupported checkpoint format: expected keys 'fc8.weight' or 'classifier.3.weight'."
        )

    if checkpoint_output_dim != len(class_names):
        print(
            "Warning: checkpoint output_dim "
            f"({checkpoint_output_dim}) != dataset classes ({len(class_names)})."
        )
        if checkpoint_output_dim > len(class_names):
            extra = [f"extra_class_{i}" for i in range(len(class_names), checkpoint_output_dim)]
            class_names = class_names + extra
        else:
            class_names = class_names[:checkpoint_output_dim]

    model.load_state_dict(checkpoint)

    metrics = {}
    # Evaluate and plot for train, validation and test datasets
    metrics["train"] = evaluate_and_plot(train_loader, model, "train", output_folder, class_names)
    metrics["validation"] = evaluate_and_plot(
        val_loader, model, "validation", output_folder, class_names
    )
    metrics["test"] = evaluate_and_plot(test_loader, model, "test", output_folder, class_names)

    # save  metrics as csv
    pd.DataFrame(metrics["train"]).to_csv(output_folder / "metrics_train.csv")
    pd.DataFrame(metrics["validation"]).to_csv(output_folder / "metrics_validation.csv")
    pd.DataFrame(metrics["test"]).to_csv(output_folder / "metrics_test.csv")
    pd.DataFrame(metrics).to_csv(output_folder / "metrics.csv")

    # Save the metrics as an image
    save_metrics_as_picture(metrics["train"], output_folder / "metrics_train.png")
    save_metrics_as_picture(metrics["validation"], output_folder / "metrics_validation.png")
    save_metrics_as_picture(metrics["test"], output_folder / "metrics_test.png")

    print("Evaluation complete!")
