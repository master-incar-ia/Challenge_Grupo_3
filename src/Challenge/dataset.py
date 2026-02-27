import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class SignosDataset(Dataset):
    def __init__(self, root=None, Mode="train", transform=None):
        base_dir = Path(root) if root else Path(__file__).resolve().parents[2] / "dataset"
        if Mode == "train":
            self.data = datasets.ImageFolder(base_dir / "train" , transform=transform)
        elif Mode == "val":
            self.data = datasets.ImageFolder(base_dir / "val", transform=transform)
        elif Mode == "test":
            self.data = datasets.ImageFolder(base_dir / "test", transform=transform)
        else:
            raise ValueError(f"Invalid mode: {Mode}. Expected 'train', 'val', or 'test'.")
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

    def plot(self, filepath):
        num_samples = min(len(self.data), 100)
        if num_samples == 0:
            raise ValueError("Cannot plot an empty dataset.")

        num_cols = min(10, num_samples)
        num_rows = math.ceil(num_samples / num_cols)

        plt.figure(figsize=(num_cols, num_rows))
        for idx in range(num_samples):
            plt.subplot(num_rows, num_cols, idx + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

            img, label = self.data[idx]
            if not torch.is_tensor(img):
                img = transforms.ToTensor()(img)
            img = img.permute(1, 2, 0)
            plt.imshow(img, cmap=plt.cm.binary)
            plt.xlabel(self.data.classes[label])
        plt.savefig(filepath)
        plt.show()
        plt.close()


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    # Data augmentation
    transform = transforms.Compose(
        [transforms.Resize((232, 232)),
            transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
    )

    dataset_train = SignosDataset(Mode="train", transform=transform)
    dataset_val = SignosDataset(Mode="val", transform=transform)
    dataset_test = SignosDataset(Mode="test", transform=transform)

    print(f"Dataset length: {len(dataset_train)}")
    print(f"First item: {dataset_train[0]}")
    dataset_train.plot(output_folder / "plot_dataset_example.png")
