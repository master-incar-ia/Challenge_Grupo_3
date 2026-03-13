import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataset import SignosDataset, build_augment_transform, build_eval_transform

# Define transformation


# Load datasets
root_dir = Path(__file__).resolve().parents[2] / "dataset"

# Dataset sin augmentation
dataset_before = SignosDataset(
    root=root_dir,
    Mode="train",
    transform=build_eval_transform(image_size=(32, 32)),
    augmented_transform=[],
    include_original=True
)

# Dataset con augmentation
dataset_after = SignosDataset(
    root=root_dir,
    Mode="train",
    transform=build_eval_transform(image_size=(32, 32)),
    augmented_transform=build_augment_transform(image_size=(32, 32)),
    include_original=True
)

# Count samples per class
class_counts_before = {}
class_counts_after = {}

for _, label in dataset_before.data.samples:
    class_name = dataset_before.classes[label]
    class_counts_before[class_name] = class_counts_before.get(class_name, 0) + 1

for _, label in dataset_after.data.samples:
    class_name = dataset_after.classes[label]
    class_counts_after[class_name] = class_counts_after.get(class_name, 0) + 1

# Multiply counts for augmented dataset (includes original + augmented versions)
for class_name in class_counts_after:
    num_aug = len(dataset_after.augmented_transforms)
    if num_aug > 0:
        class_counts_after[class_name] = class_counts_after[class_name] * (1 + num_aug)

# Prepare data for plotting
classes = sorted(class_counts_before.keys())
before_counts = [class_counts_before.get(c, 0) for c in classes]
after_counts = [class_counts_after.get(c, 0) for c in classes]

# Create comparison plot
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(classes))
width = 0.35

bars1 = ax.bar(x - width/2, before_counts, width, label='Antes de transformaciones', color='steelblue', alpha=0.8)
#bars2 = ax.bar(x + width/2, after_counts, width, label='Después de transformaciones', color='tomato', alpha=0.8)

# Add labels and title
ax.set_xlabel('Clases', fontsize=12, fontweight='bold')
ax.set_ylabel('Cantidad de muestras', fontsize=12, fontweight='bold')
ax.set_title('Distribución de clases en Dataset de Train\nAntes y Después de Transformaciones', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('class_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'class_distribution_comparison.png'")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("RESUMEN DE DISTRIBUCIÓN DE CLASES")
print("="*60)
print(f"\nTotal de muestras ANTES: {sum(before_counts)}")
print(f"Total de muestras DESPUÉS: {sum(after_counts)}")
print(f"Factor de aumento: {sum(after_counts) / sum(before_counts):.2f}x")
print("\nDetalle por clase:")
print("-"*60)
print(f"{'Clase':<8} {'Antes':<12} {'Después':<12} {'Incremento':<15}")
print("-"*60)
for cls, before, after in zip(classes, before_counts, after_counts):
    increment = after - before
    print(f"{cls:<8} {before:<12} {after:<12} +{increment:<13} ({increment/before*100:.0f}%)")
