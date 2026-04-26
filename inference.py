"""
Pneumonia Detection System — Inference & Severity Assessment
Handles image preprocessing, prediction, and opacity-based severity grading.
"""
import cv2
import numpy as np
import torch

from config import (
    DEVICE, IMAGE_SIZE, GRAYSCALE_MEAN, GRAYSCALE_STD, CLASS_NAMES,
    OPACITY_LOWER, OPACITY_UPPER, LUNG_THRESHOLD,
    SEVERITY_MILD_UPPER, SEVERITY_MODERATE_UPPER,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE
)


def preprocess_image(image_path):
    """
    Load a grayscale X-ray, resize, normalize (matching training transforms),
    and return the model-ready tensor + the original image for display.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    original = image.copy()
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    # Apply the SAME normalization used during training
    image = (image - GRAYSCALE_MEAN[0]) / GRAYSCALE_STD[0]
    image = np.expand_dims(image, axis=(0, 1))  # (1, 1, H, W)
    tensor = torch.tensor(image, dtype=torch.float32).to(DEVICE)
    return tensor, original


def predict(model, tensor):
    """Run inference and return class index, class name, and confidence."""
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    idx = pred_idx.item()
    return idx, CLASS_NAMES[idx], confidence.item()


def calculate_severity(original_img, prediction_class):
    """
    Quantify pneumonia severity using opacity ratio.

    Steps:
      1. Apply CLAHE to normalize brightness across different scanners.
      2. Threshold to isolate high-opacity (white/consolidated) regions.
      3. Compute ratio of opaque pixels to total lung area.
      4. Map ratio to Mild / Moderate / Severe.
    """
    if prediction_class == 0:
        return "No Pneumonia Detected", 0.0

    # CLAHE for adaptive contrast normalization
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE
    )
    equalized = clahe.apply(original_img)

    # High-opacity mask (consolidated/white regions)
    opacity_mask = cv2.inRange(equalized, OPACITY_LOWER, OPACITY_UPPER)

    # Approximate lung area (non-black pixels)
    _, lung_mask = cv2.threshold(equalized, LUNG_THRESHOLD, 255, cv2.THRESH_BINARY)
    total_pixels = cv2.countNonZero(lung_mask)
    opaque_pixels = cv2.countNonZero(opacity_mask)

    if total_pixels == 0:
        return "Error: Empty lung mask", 0.0

    opacity_ratio = opaque_pixels / total_pixels

    if opacity_ratio < SEVERITY_MILD_UPPER:
        severity = "Mild"
    elif opacity_ratio < SEVERITY_MODERATE_UPPER:
        severity = "Moderate"
    else:
        severity = "Severe"

    return severity, round(opacity_ratio, 4)
