# model.py

"""
Model definition for Waste Classification.
"""

import torch.nn as nn
from config import INPUT_CHANNELS, NUM_CLASSES

class WasteClassifier(nn.Module):
    """
    CNN-based model for classifying waste images into categories.
    """
    def __init__(self):
        super(WasteClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )
        

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
