# Import libraries
import time
import torch
import torch.nn as nn


# Create a basic CNN model to try classify the data
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer
        self.fc = nn.Linear(32 * 56 * 56, 10)  # Modify the output size based on your task

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x