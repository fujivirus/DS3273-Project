"""
Prediction utilities for Waste Classification.
"""

import os

import torch
import torch.nn as nn
from PIL import Image

from config import MODEL_SAVE_PATH, DATA_DIR, DEVICE
from model import WasteClassifier
from dataset import apply_transform


def load_model() -> nn.Module:
    """
    Loads the best saved model from checkpoint.
    """
    model = WasteClassifier()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def classify_waste(image_paths: list[str]) -> list[int]:
    """
    Classify a batch of images and return predicted labels.

    Args:
        image_paths: List of paths (relative to data/images/)
        device: torch device (cpu/cuda)
        num_classes: total number of classes

    Returns:
        List of predicted labels
    """
    model = load_model()

    predictions = []

    for img_rel_path in image_paths:
        img_path = os.path.join(DATA_DIR, img_rel_path)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        image = apply_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            _, pred = outputs.max(1)
            predictions.append(pred.item())

    return predictions


def evaluate_testset(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluate model on test set.

    Returns:
        (test_loss, test_accuracy)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
