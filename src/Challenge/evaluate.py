from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataset import CIFAR10Dataset
from .model import VGG, ConvolutionalNeuralNetwork


def evaluate_and_plot(loader, model, dataset_name, output_folder):
    model.eval()
    all_inputs = []
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            all_inputs.append(inputs.numpy())
            all_outputs.append(outputs.numpy())
            all_targets.append(targets.numpy())

    all_inputs = np.concatenate(all_inputs)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    # Lets plot the confusion matrix as a heatmap
    indexes_of_outputs = []
    for elements in all_outputs:
        indexes_of_outputs.append(np.argmax(elements))
    indexes_of_outputs = np.array(indexes_of_outputs)

    # Classes of CIFAR-10 dataset
    cifar10_classes = [
        "Avión",
        "Coche",
        "Pájaro",
        "Gato",
        "Ciervo",
        "Perro",
        "Rana",
        "Caballo",
        "Barco",
        "Camión",
    ]

    # Set the confusion matrix
    map = confusion_matrix(all_targets, indexes_of_outputs)

    # Plot and save the confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        map,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cifar10_classes,
        yticklabels=cifar10_classes,
    )

    plt.xlabel("Predicción")
    plt.ylabel("Realidad (Target)")
    plt.title("Matriz de Confusión - CIFAR-10")
    plt.savefig(output_folder / f"confusion_matrix_{dataset_name}.png")
    plt.show()

    # Let's obtain the accuracy, precision, recall and F1 score for the dataset
    predictions = np.zeros(len(all_targets))
    for i in range(len(all_targets)):
        if all_targets[i] == all_outputs[i].argmax():
            predictions[i] = 1
        else:
            predictions[i] = 0

    # Calculate accuracy
    accuracy = sum(predictions) / len(predictions)
    print(f"Accuracy: {accuracy:.4f}")

    prec, rec, f1, support = precision_recall_fscore_support(
        all_targets, indexes_of_outputs, average=None
    )

    print(f"{'CLASE':<10} {'PRECISION':<10} {'RECALL':<10} {'F1-SCORE':<10} {'CANTIDAD (Support)'}")
    print("-" * 60)

    for i in range(10):
        print(
            f"{cifar10_classes[i]:<10} {prec[i]:.2f}       {rec[i]:.2f}       {f1[i]:.2f}       {support[i]}"
        )

    metrics = {
        "Accuracy": accuracy,
        "Clase": [
            "Avión",
            "Coche",
            "Pájaro",
            "Gato",
            "Ciervo",
            "Perro",
            "Rana",
            "Caballo",
            "Barco",
            "Camión",
        ],
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


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)
    # Set the seed for reproducibility
    torch.manual_seed(42)
    # Data augmentation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
    )

    # Cargas el bloque de 50k
    full_train_data = CIFAR10Dataset("./data", train=True, transform=transform)
    # Divides ese bloque en dos (90% entrenamiento, 10% validación interna)
    train_subset, val_subset = random_split(full_train_data, [45000, 5000])

    # Cargas el bloque de 10k de test (con transform limpia, sin augmentation)
    test_dataset = CIFAR10Dataset("./data", train=False, transform=transform)

    # Create DataLoaders for the datasets
    train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Load the best model weights
    model = VGG(output_dim=10)
    model.load_state_dict(torch.load(output_folder / "best_model.pth"))

    metrics = {}
    # Evaluate and plot for train, validation and test datasets
    metrics["train"] = evaluate_and_plot(train_loader, model, "train", output_folder)
    metrics["validation"] = evaluate_and_plot(val_loader, model, "validation", output_folder)
    metrics["test"] = evaluate_and_plot(test_loader, model, "test", output_folder)

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
