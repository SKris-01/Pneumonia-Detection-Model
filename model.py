"""
Pneumonia Detection System — Model Architecture & Training
ResNet-50 fine-tuned for grayscale chest X-ray classification.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from config import DEVICE, NUM_CLASSES, LEARNING_RATE, EPOCHS, MODEL_SAVE_PATH


def build_model():
    """
    Build a ResNet-50 adapted for 1-channel grayscale input.

    Key fix: Instead of randomly initializing conv1, we average the
    pretrained 3-channel RGB weights into a single channel — preserving
    learned edge/texture features from ImageNet.
    """
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Adapt conv1: average pretrained RGB weights → 1 grayscale channel
    pretrained_w = model.conv1.weight.data  # (64, 3, 7, 7)
    new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    new_conv1.weight.data = pretrained_w.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
    model.conv1 = new_conv1

    # Replace final FC for binary classification
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model.to(DEVICE)


def train_model(model, train_loader, val_loader):
    """Train with validation tracking per epoch. Returns training history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # ── Validate ──
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                _, preds = torch.max(model(inputs), 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        history['val_acc'].append(val_acc)
        scheduler.step()

        print(f"  Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    return history


def load_trained_model():
    """Load a previously saved model checkpoint."""
    model = build_model()
    model.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model
