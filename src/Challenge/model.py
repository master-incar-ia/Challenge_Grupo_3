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
        self.activation3 = nn.Softmax(dim=1)

    def forward(self, x, use_activation=True):
        x1 = self.con2D_1(x)
        x2 = self.activation1(x1)
        x3 = self.max_pooling1(x2)
        x4 = self.con2D_2(x3)
        x5 = self.activation2(x4)
        x6 = self.max_pooling2(x5)
        x7 = self.flatten(x6)
        x8 = self.Final_Layer3(x7)
        x9 = self.activation3(x8)
        return x9


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
        self.softmax = nn.Softmax(dim=1)

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
        x38 = self.softmax(x37)

        return x38


if __name__ == "__main__":
    model = ConvolutionalNeuralNetwork(output_dim=10)
    print(model)
    x = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10
    print(model(x, use_activation=True))
    pass
