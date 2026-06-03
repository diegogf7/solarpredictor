import torch
import torch.nn as nn

class SolarCNN(nn.Module):
    def __init__(self):
        super(SolarCNN, self).__init__()

        self.features = nn.Sequential(
            #one layer
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #second layer
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #third layer
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #fourth layer
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14 , 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 1)
        )

    def forward(self, k):
        k = self.features(k)
        k = self.classifier(k)
        return k