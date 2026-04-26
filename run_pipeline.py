"""
Pneumonia Detection — Full Pipeline Runner
Trains the model, runs inference on a pneumonia image,
performs grid analysis, Grad-CAM, final report, and evaluation.
"""
import os
import sys
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ─── Our modules ────────────────────────────────────────
from config import DATA_DIR, DEVICE, CLASS_NAMES
from data_loader import load_datasets
from model import build_model, train_model
from inference import preprocess_image, predict, calculate_severity
from visualization import (
    plot_training_history, show_lung_roi, analyze_grid_regions,
    generate_gradcam, show_gradcam_panel, show_final_report
)

# Create output directory for saved figures
OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(name):
    """Save the current matplotlib figure to results folder."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {path}")
    plt.close()


def main():
    print("=" * 60)
    print("  🫁 PNEUMONIA DETECTION WITH SEVERITY ASSESSMENT")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATA_DIR}")
    print()

    # ─── STEP 1: Load Dataset ────────────────────────────
    print("─" * 60)
    print("STEP 1: Loading Binary Dataset (NORMAL vs PNEUMONIA)")
    print("─" * 60)
    train_loader, val_loader, test_loader, class_names = load_datasets()
    print()

    # ─── STEP 2: Train Model ────────────────────────────
    print("─" * 60)
    print("STEP 2: Training ResNet-50")
    print("─" * 60)
    model = build_model()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    history = train_model(model, train_loader, val_loader)
    plot_training_history(history)
    save_fig("01_training_curves")
    print()

    # ─── STEP 3: Pick Pneumonia Image → Inference + Severity ─
    print("─" * 60)
    print("STEP 3: Inference & Severity Assessment")
    print("─" * 60)
    pneumonia_dir = os.path.join(DATA_DIR, 'test', 'PNEUMONIA')
    pneumonia_images = sorted(glob.glob(os.path.join(pneumonia_dir, '*')))
    print(f"  Found {len(pneumonia_images)} pneumonia test images")

    img_path = pneumonia_images[0]
    print(f"  Selected: {os.path.basename(img_path)}")

    tensor, original_img = preprocess_image(img_path)
    pred_idx, diagnosis, confidence = predict(model, tensor)
    severity_label, score = calculate_severity(original_img, pred_idx)

    print(f"\n  Diagnosis:  {diagnosis}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Severity:   {severity_label}")
    print(f"  Opacity:    {score:.4f}")

    show_lung_roi(original_img)
    save_fig("02_lung_roi")
    print()

    # ─── STEP 4: Grid-Line Region Analysis ───────────────
    print("─" * 60)
    print("STEP 4: Grid-Line Region Analysis (4×4)")
    print("─" * 60)
    grid_scores = analyze_grid_regions(original_img, grid_size=4)
    save_fig("03_grid_analysis")
    print()

    # ─── STEP 5: Grad-CAM Visualization ──────────────────
    print("─" * 60)
    print("STEP 5: Grad-CAM Visualization")
    print("─" * 60)
    visualization, grayscale_cam = generate_gradcam(model, tensor, original_img)
    show_gradcam_panel(original_img, visualization, grayscale_cam)
    save_fig("04_gradcam")
    print("   Grad-CAM generated successfully")
    print()

    # ─── STEP 6: Final Report ────────────────────────────
    print("─" * 60)
    print("STEP 6: Final Diagnostic Report")
    print("─" * 60)
    show_final_report(
        original_img, visualization,
        severity_label, score,
        diagnosis, confidence,
        grid_scores=grid_scores
    )
    save_fig("05_final_report")
    print()

    # ─── STEP 7: Model Evaluation ────────────────────────
    print("─" * 60)
    print("STEP 7: Model Evaluation on Full Test Set")
    print("─" * 60)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            _, preds = torch.max(model(inputs), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n  CLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig("06_confusion_matrix")

    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\n  ✅ Overall Test Accuracy: {accuracy:.2%}")

    # ─── Done ────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  ✅ ALL RESULTS SAVED TO: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
