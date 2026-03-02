import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

MEAN = [0.50956494, 0.50055039, 0.49491626] 
STD = [0.07214967, 0.09587376, 0.11140345]

class SignosDataset(Dataset):
    def __init__(
        self,
        root=None,
        Mode="train",
        transform=None,
        augmented_transform=None,
        include_original=True,
    ):
        base_dir = Path(root) if root else Path(__file__).resolve().parents[2] / "dataset"
        self.transform = transform
        self.augmented_transform = augmented_transform
        self.include_original = include_original

        if Mode == "train":
            self.data = datasets.ImageFolder(base_dir / "train", transform=None)
        elif Mode == "val":
            self.data = datasets.ImageFolder(base_dir / "val", transform=None)
        elif Mode == "test":
            self.data = datasets.ImageFolder(base_dir / "test", transform=None)
        else:
            raise ValueError(f"Invalid mode: {Mode}. Expected 'train', 'val', or 'test'.")

        self.classes = self.data.classes
       
    def __len__(self):
        if self.augmented_transform is not None and self.include_original:
            return len(self.data) * 2
        return len(self.data)

    def __getitem__(self, idx):
        base_len = len(self.data)

        use_augmented = False
        if self.augmented_transform is not None and self.include_original:
            if idx >= base_len:
                use_augmented = True
                idx = idx - base_len
        elif self.augmented_transform is not None and not self.include_original:
            use_augmented = True

        image_path, label = self.data.samples[idx]
        image = self.data.loader(image_path)

        if use_augmented:
            if self.augmented_transform is not None:
                image = self.augmented_transform(image)
        else:
            if self.transform is not None:
                image = self.transform(image)

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


class AddGaussianNoise:
    def __init__(self, std=0.03):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


def build_eval_transform(image_size=(32, 32)):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def build_augment_transform(image_size=(32, 32)):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.2,
                        hue=0.03,
                    )
                ],
                p=0.7,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.RandomApply([AddGaussianNoise(std=0.03)], p=0.3),
            transforms.Normalize(MEAN, STD),
        ]
    )

if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    eval_transform = build_eval_transform(image_size=(32, 32))
    augment_transform = build_augment_transform(image_size=(32, 32))

    dataset_train = SignosDataset(
        Mode="train",
        transform=eval_transform,
        augmented_transform=augment_transform,
        include_original=True,
    )
    dataset_val = SignosDataset(Mode="val", transform=eval_transform)
    dataset_test = SignosDataset(Mode="test", transform=eval_transform)

    print(f"Dataset length: {len(dataset_train)}")
    print(f"First item: {dataset_train[0]}")
    dataset_val.plot(output_folder / "plot_dataset_example2.png")
