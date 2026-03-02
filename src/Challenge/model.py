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


class VGG(nn.Module):
    def __init__(self, output_dim=1000, in_channels=3):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc6 = nn.Linear(8192, 8192)
        self.relu14 = nn.ReLU()
        self.fc7 = nn.Linear(8192, 512)
        self.relu15 = nn.ReLU()
        self.fc8 = nn.Linear(512, output_dim)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.relu1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu2(x3)
        x5 = self.pool1(x4)

        x6 = self.conv2_1(x5)
        x7 = self.relu3(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu4(x8)
        x10 = self.pool2(x9)

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
        self.convolution = nn.Conv2d(3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.convolution3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.activation3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)
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
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = self.relu2(out)
        return out


class SimpleResNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, output_dim)

    def _make_layer(self, in_channels, out_channels, stride):
        return ResidualBlock(in_channels, out_channels, stride)

    def forward(self, x, use_activation=False):
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


class VGG_mask(nn.Module):
    def __init__(self, output_dim=26, in_channels=3):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc6 = nn.Linear(8192, 8192)
        self.relu5 = nn.ReLU()
        self.fc7 = nn.Linear(8192, 512)
        self.relu6 = nn.ReLU()
        self.fc8 = nn.Linear(512, output_dim)

        self.mask_up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.mask_conv1 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mask_up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.mask_conv2 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mask_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.relu1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu2(x3)
        x5 = self.pool1(x4)

        x6 = self.conv2_1(x5)
        x7 = self.relu3(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu4(x8)
        x10 = self.pool2(x9)

        cls = self.flatten(x10)
        cls = self.fc6(cls)
        cls = self.relu5(cls)
        cls = self.fc7(cls)
        cls = self.relu6(cls)
        class_logits = self.fc8(cls)

        mask = self.mask_up1(x10)
        mask = torch.cat([mask, x9], dim=1)
        mask = self.mask_conv1(mask)
        mask = self.mask_up2(mask)
        mask = torch.cat([mask, x4], dim=1)
        mask = self.mask_conv2(mask)
        mask_logits = self.mask_head(mask)

        return class_logits, mask_logits






  

if __name__ == "__main__":
    model = VGG_mask(output_dim=26, in_channels=3)
    print(model)
    image = torch.randn(1, 3, 32, 32)
    class_logits, mask_logits = model(image)
    print(class_logits.shape, mask_logits.shape)
