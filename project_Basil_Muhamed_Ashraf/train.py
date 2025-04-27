"""
Training loop for Waste Classification model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import LEARNING_RATE, MODEL_SAVE_PATH, DEVICE


def calculate_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes the accuracy of predictions.
    """
    _, preds_max = preds.max(1)
    correct = (preds_max == labels).sum().item()
    return correct / labels.size(0)


def train_model(
    model: nn.Module,
    num_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
) -> dict[str, list]:
    """
    Runs the training and validation loops.
    Prints training/validation loss & accuracy at each epoch.
    Saves the model.
    """
    optimizer_ = optimizer(model.parameters(), lr=LEARNING_RATE)
    criterion = loss_fn

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer_.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_.step()

            running_loss += loss.item() * images.size(0)
            running_acc += calculate_accuracy(outputs, labels) * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore[arg-type]
        epoch_acc = running_acc / len(train_loader.dataset)  # type: ignore[arg-type]

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_acc += calculate_accuracy(outputs, labels) * images.size(0)

        val_loss /= len(val_loader.dataset)  # type: ignore[arg-type]
        val_acc /= len(val_loader.dataset)  # type: ignore[arg-type]

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}]: "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    return history


def train_model_no_val(
    model: nn.Module,
    num_epochs: int,
    train_loader: DataLoader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
) -> dict[str, list]:
    """
    Runs the training loops.
    Prints training loss at each epoch.
    Saves the model.
    """
    optimizer_ = optimizer(model.parameters(), lr=LEARNING_RATE)
    criterion = loss_fn

    history = {
        "train_loss": [],
        "train_acc": [],
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer_.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore[arg-type]

        history["train_loss"].append(epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss: {epoch_loss:.4f}")

        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    return history
