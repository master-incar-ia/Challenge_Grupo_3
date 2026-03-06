# calcular_stats.py
import torch
from torch.utils.data import DataLoader
from dataset import SignosDataset, build_eval_transform

def main():
    transform = build_eval_transform(image_size=(32, 32))  # sin normalizar
    dataset = SignosDataset(Mode="train", transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)  # workers > 0

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        batch_pixels = images.shape[0] * images.shape[2] * images.shape[3]
        total_pixels += batch_pixels
        mean += images.mean(dim=[0, 2, 3]) * batch_pixels
        std += images.std(dim=[0, 2, 3]) * batch_pixels

    mean /= total_pixels
    std /= total_pixels

    print(f"Media calculada: {mean.tolist()}")
    print(f"Desviación calculada: {std.tolist()}")

if __name__ == '__main__':
    main()