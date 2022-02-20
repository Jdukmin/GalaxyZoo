#시스템 모듈
import os
import copy
import numpy as np
import time

#Pytorch 모듈
import torch
import torch.nn as nn
import torch.nn.functional as F

#신경망을 정의하는 블럭
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5) # input channels, output channels, kernel size
        self.pool = nn.MaxPool2d(2, 2, 1)  # kernel size, stride, padding = 0 (default)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 2 * 2, 192) # input features, output features
        self.fc2 = nn.Linear(192, 128)
        self.fc3 = nn.Linear(128, 84)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class SanderDielemanNet(nn.Module) :
    def __init__(self, num_classes=37) :
        super(SanderDielemanNet, self).__init__()
        # Convolutional and MaxPool layers
        self.conv1 = nn.Conv2d(3, 32, 6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dense layers
        self.fc1 = nn.Linear(128*2*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Convolutional and MaxPool layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        # Dense layers
        x = x.view(-1, 128*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return(x)