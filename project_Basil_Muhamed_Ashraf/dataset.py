# dataset.py

"""
Dataset and DataLoader utilities for Waste Classification.
"""

import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import RESIZE_X, RESIZE_Y, DATA_DIR, BATCH_SIZE, NUM_WORKERS

def apply_transform(image: Image.Image) -> torch.Tensor:
    image = transforms.Resize((RESIZE_X, RESIZE_Y))(image)
    image_t = transforms.ToTensor()(image)
    image_t = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(image_t)
    return image_t

class WasteDataset(Dataset[tuple[torch.Tensor, int]]):
    """
    Custom Dataset for loading waste classification images.
    """

    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = root_dir
        self.split = split.lower()
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels: list[int] = []

        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        """
        Prepare dataset by scanning directories and splitting into train/val/test.
        """
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)

            for subfolder in ["default", "real_world"]:
                subfolder_dir = os.path.join(class_dir, subfolder)

                if not os.path.isdir(subfolder_dir):
                    continue  # Skip if missing

                image_names = os.listdir(subfolder_dir)
                random.shuffle(image_names)

                n_total = len(image_names)
                n_train = int(0.6 * n_total)
                n_val = int(0.2 * n_total)

                if self.split == "train":
                    selected_images = image_names[:n_train]
                elif self.split == "val":
                    selected_images = image_names[n_train : n_train + n_val]
                else:  # "test"
                    selected_images = image_names[n_train + n_val :]

                for img_name in selected_images:
                    self.image_paths.append(os.path.join(subfolder_dir, img_name))
                    self.labels.append(idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        image = apply_transform(image)

        return image, label


class WasteDataLoader:
    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size

    def get_loaders(self):
        train_dataset = WasteDataset(DATA_DIR, split="train")
        val_dataset = WasteDataset(DATA_DIR, split="val")
        test_dataset = WasteDataset(DATA_DIR, split="test")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        return train_loader, val_loader, test_loader
