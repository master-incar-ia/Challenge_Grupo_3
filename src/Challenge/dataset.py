import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import skimage as ski
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
        self.include_original = include_original

        if augmented_transform is None:
            self.augmented_transforms = []
        elif isinstance(augmented_transform, (list, tuple)):
            self.augmented_transforms = list(augmented_transform)
        else:
            self.augmented_transforms = [augmented_transform]

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
        num_aug = len(self.augmented_transforms)
        if num_aug > 0 and self.include_original:
            return len(self.data) * (1 + num_aug)
        if num_aug > 0 and not self.include_original:
            return len(self.data) * num_aug
        return len(self.data)

    def __getitem__(self, idx):
        base_len = len(self.data)
        num_aug = len(self.augmented_transforms)

        use_augmented = False
        selected_augment = None

        if num_aug > 0:
            variant_idx = idx // base_len
            sample_idx = idx % base_len

            if self.include_original:
                if variant_idx == 0:
                    use_augmented = False
                else:
                    use_augmented = True
                    selected_augment = self.augmented_transforms[variant_idx - 1]
            else:
                use_augmented = True
                selected_augment = self.augmented_transforms[variant_idx]

            idx = sample_idx

        image_path, label = self.data.samples[idx]
        image = self.data.loader(image_path)

        if use_augmented:
            if selected_augment is not None:
                image = selected_augment(image)
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


def _tensor_to_display_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a transformed tensor to an RGB image in [0, 1] for plotting/saving."""
    image = tensor.detach().cpu().float()
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 3:
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        image = image * std + mean
    return image.clamp(0.0, 1.0)


def save_transformations_preview(
    dataset: SignosDataset,
    filepath,
    sample_idx: int = 0,
    include_original: bool = True,
):
    """Save a grid image with all transformations applied to one dataset sample."""
    if len(dataset.data) == 0:
        raise ValueError("Cannot preview transformations for an empty dataset.")
    if sample_idx < 0 or sample_idx >= len(dataset.data):
        raise IndexError(f"sample_idx {sample_idx} is out of range for base dataset length {len(dataset.data)}")

    image_path, label = dataset.data.samples[sample_idx]
    original_image = dataset.data.loader(image_path)
    class_name = dataset.classes[label]

    transformed_images = []
    titles = []

    if include_original:
        if dataset.transform is not None:
            transformed_images.append(_tensor_to_display_image(dataset.transform(original_image)))
            titles.append("base_transform")
        else:
            transformed_images.append(transforms.ToTensor()(original_image))
            titles.append("original")

    for idx, aug_transform in enumerate(dataset.augmented_transforms, start=1):
        transformed_images.append(_tensor_to_display_image(aug_transform(original_image)))
        titles.append(f"aug_{idx}")

    if not transformed_images:
        transformed_images.append(transforms.ToTensor()(original_image))
        titles.append("original")

    num_images = len(transformed_images)
    num_cols = min(4, num_images)
    num_rows = math.ceil(num_images / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols, 3.5 * num_rows))
    if not isinstance(axes, (list, tuple)):
        axes = [axes] if num_images == 1 else axes.flatten()

    for i in range(num_rows * num_cols):
        ax = axes[i]
        ax.axis("off")
        if i < num_images:
            img = transformed_images[i].permute(1, 2, 0)
            if transformed_images[i].shape[0] == 1:
                ax.imshow(img.squeeze(-1), cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(titles[i], fontsize=9)

    fig.suptitle(f"Class: {class_name} | sample_idx={sample_idx}")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)




def build_eval_transform(image_size=(32, 32)):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def _hsv_saturation_mask(tensor: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    rgb = tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    hsv = ski.color.rgb2hsv(rgb)
    saturation = torch.from_numpy(hsv[..., 1]).unsqueeze(0).float()
    return (saturation > threshold).float()


def _ensure_float_tensor(image) -> torch.Tensor:
    if torch.is_tensor(image):
        tensor = image
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)

        tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    return transforms.ToTensor()(image)


def MaskTransform(image_size=(32, 32)):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.Lambda(_ensure_float_tensor),
            transforms.Lambda(lambda x: _hsv_saturation_mask(x, threshold=0.15)),
        ]
    )
def build_augment_transform(image_size=(32, 32)):
    return [
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomRotation(degrees=(-8, 8)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomRotation(degrees=(-15, 15)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.1,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.25,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                AddGaussianNoise(std=0.02),
                transforms.Normalize(MEAN, STD),
            ]
        ),

    ]

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
    #dataset_val.plot(output_folder / "plot_dataset_example2.png")

    # Save one image with all transforms applied to the same sample.
    save_transformations_preview(
        dataset=dataset_train,
        filepath=output_folder / "all_transformations_preview.png",
        sample_idx=0,
        include_original=True,
    )

    first_sample = _tensor_to_display_image(dataset_train[0][0])
    plt.imsave(output_folder / "first_sample_train.png", first_sample.permute(1, 2, 0).numpy())

    mask_transform = MaskTransform(image_size=(32, 32))
    dataset_val_raw = SignosDataset(Mode="val", transform=build_augment_transform)
    sample_pil, sample_label = dataset_val_raw.data[6]
    sample_rgb = transforms.ToTensor()(sample_pil)
    sample_mask = mask_transform(sample_pil)
    print(
        f"MaskTransform test -> class: {dataset_val_raw.classes[sample_label]}, "
        f"rgb shape: {tuple(sample_rgb.shape)}, mask shape: {tuple(sample_mask.shape)}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(sample_rgb.permute(1, 2, 0))
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(sample_mask.squeeze(0), cmap="gray")
    axes[1].set_title("Mask (S > 0.15)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_folder / "mask_transform_test.png")
    plt.close(fig)
    
