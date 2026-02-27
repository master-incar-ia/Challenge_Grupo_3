import torch
import torch.nn as nn


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.con2D_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.activation1 = nn.ReLU()
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.con2D_2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0
        )
        self.activation2 = nn.ReLU()
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.Final_Layer3 = nn.Linear(32 * 6 * 6, output_dim)

    def forward(self, x, use_activation=True):
        x1 = self.con2D_1(x)
        x2 = self.activation1(x1)
        x3 = self.max_pooling1(x2)
        x4 = self.con2D_2(x3)
        x5 = self.activation2(x4)
        x6 = self.max_pooling2(x5)
        x7 = self.flatten(x6)
        x8 = self.Final_Layer3(x7)
        return x8


# Definimos el modelo VGG_16, siguiendo la página 54 de la diapositiva "Convolutional Neural Networks (CNNs)".
class VGG(nn.Module):
    def __init__(self, output_dim=1000):
        super().__init__()

        # 1º Bloque de VGG:
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2º Bloque de VGG:
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #  Ultima capa del VGG:
        self.flatten = nn.Flatten()
        # Para CIFAR-10 (32x32): 512 * 2 * 2 = 2048 (en lugar de 25088 para ImageNet 224x224)
        self.fc6 = nn.Linear(8192, 8192)
        self.relu14 = nn.ReLU()
        self.fc7 = nn.Linear(8192, 512)
        self.relu15 = nn.ReLU()
        self.fc8 = nn.Linear(512, output_dim)

    def forward(self, x):
        # Bloque 1
        x1 = self.conv1_1(x)
        x2 = self.relu1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu2(x3)
        x5 = self.pool1(x4)

        # Bloque 2
        x6 = self.conv2_1(x5)
        x7 = self.relu3(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu4(x8)
        x10 = self.pool2(x9)

        # Clasificador
        x32 = self.flatten(x10)
        x33 = self.fc6(x32)
        x34 = self.relu14(x33)
        x35 = self.fc7(x34)
        x36 = self.relu15(x35)
        x37 = self.fc8(x36)
        return x37
    
class ConvolutionalNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Primera capa convolucional
        self.convolution = nn.Conv2d(3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        
        # Segunda capa convolucional
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.ReLU()
        
        # Primer maxpool
        self.maxpool1 = nn.MaxPool2d(2)
        
        # Tercera capa convolucional
        self.convolution3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.activation3 = nn.ReLU()
        
        # Segundo maxpool
        self.maxpool2 = nn.MaxPool2d(2)
        
        # Capas fully connected
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)
        
        # Cálculo de dimensiones para entrada 32x32:
        # Entrada 32x32
        # Después de conv1 y conv2 (padding=1) → 32x32
        # Después de maxpool1 (2) → 16x16
        # Después de conv3 (padding=1) → 16x16
        # Después de maxpool2 (2) → 8x8
        # Con 64 canales: 64 * 8 * 8 = 4096
        
        self.linear = nn.Linear(64 * 8 * 8, output_dim)
        self.final_activation = nn.Softmax(dim=1)
        
    def forward(self, x, use_activation=True):
        x = self.activation(self.convolution(x))
        x = self.activation2(self.convolution2(x))
        x = self.maxpool1(x)
        x = self.activation3(self.convolution3(x))
        x = self.maxpool2(x)
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        
        if use_activation:
            y = self.final_activation(x)
        else:
            y = x
        return y
    
class ResidualBlock(nn.Module):
    """
    Bloque residual básico sin BatchNorm.
    Dos convoluciones 3x3 con ReLU y atajo (shortcut) convolucional cuando cambian las dimensiones.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Primera convolución (puede reducir resolución si stride > 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        
        # Segunda convolución (siempre stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        
        # Atajo: si cambia el número de canales o la resolución, usamos una conv 1x1
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                      stride=stride, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + self.shortcut(x)  # conexión residual
        out = self.relu2(out)
        return out

class SimpleResNet(nn.Module):
    """
    ResNet muy simple para imágenes pequeñas (32x32).
    - Capa inicial: Conv 3x3, 64 canales.
    - 4 bloques residuales (un bloque por etapa) con increasing channels: 64 → 128 → 256 → 512.
    - Reducción espacial en los bloques 2, 3 y 4 (stride=2).
    - Global Average Pooling + capa lineal final.
    """
    def __init__(self, output_dim):
        super().__init__()
        # Primera capa convolucional
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        
        # Bloques residuales
        self.layer1 = self._make_layer(64, 64, stride=1)   # 32x32
        self.layer2 = self._make_layer(64, 128, stride=2)  # 16x16
        self.layer3 = self._make_layer(128, 256, stride=2) # 8x8
        self.layer4 = self._make_layer(256, 512, stride=2) # 4x4
        
        # Clasificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, output_dim)  # nombre 'linear' para compatibilidad con evaluate.py

    def _make_layer(self, in_channels, out_channels, stride):
        """Crea un bloque residual (una sola capa en nuestra versión simple)."""
        return ResidualBlock(in_channels, out_channels, stride)

    def forward(self, x, use_activation=False):
        """
        use_activation: si es True aplica softmax (no usado en entrenamiento/evaluación normal,
                        pero se mantiene por compatibilidad con otros modelos).
        """
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        
        if use_activation:
            return torch.softmax(x, dim=1)
        return x

class MultiTaskVGG(nn.Module):
    def __init__(self, output_dim=1000):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.relu1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu2(x3)
        x5 = self.pool1(x4)

        x6 = self.conv2_1(x5)
        x7 = self.relu3(x6)
        x8 = self.conv2_2(x7)
        features = self.relu4(x8)
        pooled = self.pool2(features)

        class_logits = self.classifier(pooled)
        mask_logits = self.segmentation_head(pooled)
        return class_logits, mask_logits


if __name__ == "__main__":
    model = MultiTaskVGG(output_dim=10)
    print(model)
    x = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10
    print(model(x))
    pass
