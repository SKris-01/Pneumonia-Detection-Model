"""
Pneumonia Detection System — Data Loading & Preprocessing
Handles dataset loading, transforms, and train/val/test splitting.
"""
import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

from config import (
    DATA_DIR, IMAGE_SIZE, GRAYSCALE_MEAN, GRAYSCALE_STD,
    BATCH_SIZE, VAL_SPLIT_RATIO
)


def get_transforms():
    """Return train and test transform pipelines."""
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(GRAYSCALE_MEAN, GRAYSCALE_STD),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(GRAYSCALE_MEAN, GRAYSCALE_STD),
    ])
    return train_tf, test_tf


def load_datasets():
    """Load train/val/test datasets and return DataLoaders + class names."""
    train_tf, test_tf = get_transforms()

    full_train = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'train'), transform=train_tf
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'test'), transform=test_tf
    )

    # Train / Validation split
    val_size = int(VAL_SPLIT_RATIO * len(full_train))
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = full_train.classes  # e.g. ['NORMAL', 'PNEUMONIA']

    print(f"Training samples:   {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Testing samples:    {len(test_dataset)}")
    print(f"Classes:            {class_names}")

    return train_loader, val_loader, test_loader, class_names
