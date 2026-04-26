"""
Pneumonia Detection System — Visualization
Grid-line analysis, Grad-CAM heatmaps, severity gauge, and training curves.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE


# ═══════════════════════════════════════════════════════════
#  Training History
# ═══════════════════════════════════════════════════════════

def plot_training_history(history):
    """Plot training loss and validation accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['val_acc'], 'g-o', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════
#  Lung ROI Display
# ═══════════════════════════════════════════════════════════

def show_lung_roi(original_img):
    """Display original X-ray alongside the thresholded lung mask."""
    _, mask = cv2.threshold(original_img, 50, 255, cv2.THRESH_BINARY)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title("Original X-Ray")
    ax1.axis('off')

    ax2.imshow(mask, cmap='gray')
    ax2.set_title("Approximate Lung Mask")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════
#  Grid-Line Region Analysis
# ═══════════════════════════════════════════════════════════

def analyze_grid_regions(original_img, grid_size=4):
    """
    Overlay a grid on the X-ray and analyze opacity per region.

    Divides the image into grid_size x grid_size cells, computes
    mean intensity in each cell, and color-codes them:
      - Red    = High opacity (likely consolidation)
      - Orange = Medium opacity
      - Green  = Low opacity (clear lung tissue)

    Returns the grid scores matrix and displays the annotated image.
    """
    h, w = original_img.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    # CLAHE equalization for consistent analysis
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    equalized = clahe.apply(original_img)

    grid_scores = np.zeros((grid_size, grid_size))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: Grid overlay on X-ray ──
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Grid Region Analysis", fontsize=13, fontweight='bold')

    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = equalized[y1:y2, x1:x2]

            # Opacity score = fraction of bright pixels (>150) in cell
            bright_pixels = np.sum(cell > 150)
            total_pixels = cell.size
            opacity = bright_pixels / total_pixels if total_pixels > 0 else 0
            grid_scores[i, j] = opacity

            # Color coding
            if opacity > 0.35:
                color = 'red'
                alpha = 0.35
            elif opacity > 0.15:
                color = 'orange'
                alpha = 0.25
            else:
                color = 'lime'
                alpha = 0.15

            rect = patches.Rectangle(
                (x1, y1), cell_w, cell_h,
                linewidth=1.5, edgecolor='white',
                facecolor=color, alpha=alpha
            )
            axes[0].add_patch(rect)
            axes[0].text(
                x1 + cell_w / 2, y1 + cell_h / 2,
                f"{opacity:.0%}",
                color='white', fontsize=8, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)
            )

    # Draw grid lines explicitly
    for i in range(1, grid_size):
        axes[0].axhline(y=i * cell_h, color='white', linewidth=1, alpha=0.7)
        axes[0].axvline(x=i * cell_w, color='white', linewidth=1, alpha=0.7)
    axes[0].axis('off')

    # ── Panel 2: Heatmap of grid scores ──
    im = axes[1].imshow(grid_scores, cmap='YlOrRd', vmin=0, vmax=0.6,
                        interpolation='nearest')
    axes[1].set_title("Region Opacity Heatmap", fontsize=13, fontweight='bold')
    for i in range(grid_size):
        for j in range(grid_size):
            axes[1].text(j, i, f"{grid_scores[i, j]:.0%}",
                         ha='center', va='center', fontsize=11,
                         color='black' if grid_scores[i, j] < 0.3 else 'white',
                         fontweight='bold')
    axes[1].set_xticks(range(grid_size))
    axes[1].set_yticks(range(grid_size))
    axes[1].set_xticklabels([f"C{j+1}" for j in range(grid_size)])
    axes[1].set_yticklabels([f"R{i+1}" for i in range(grid_size)])
    plt.colorbar(im, ax=axes[1], label='Opacity Score', shrink=0.8)

    plt.tight_layout()
    plt.show()

    # Print summary
    max_idx = np.unravel_index(np.argmax(grid_scores), grid_scores.shape)
    print(f"\n📊 Grid Analysis ({grid_size}×{grid_size}):")
    print(f"   Highest opacity region: R{max_idx[0]+1}-C{max_idx[1]+1} "
          f"({grid_scores[max_idx]:.1%})")
    print(f"   Mean opacity across grid: {grid_scores.mean():.1%}")
    high_regions = np.sum(grid_scores > 0.35)
    print(f"   High-opacity regions (>35%): {high_regions}/{grid_size**2}")

    return grid_scores


# ═══════════════════════════════════════════════════════════
#  Grad-CAM
# ═══════════════════════════════════════════════════════════

def generate_gradcam(model, tensor, original_img):
    """
    Generate Grad-CAM heatmap overlay.

    Key: Input tensor stays 1-channel (matching model.conv1).
    Only the display overlay is converted to RGB.
    """
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # 1-channel tensor — matches the model's conv1
    grayscale_cam = cam(input_tensor=tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # RGB conversion only for display overlay
    resized = cv2.resize(original_img, (224, 224))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization, grayscale_cam


def show_gradcam_panel(original_img, visualization, grayscale_cam):
    """Display Grad-CAM with original, heatmap overlay, and raw activation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original X-Ray")
    axes[0].axis('off')

    axes[1].imshow(visualization)
    axes[1].set_title("Grad-CAM Overlay\n(Red = Model Focus)")
    axes[1].axis('off')

    im = axes[2].imshow(grayscale_cam, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title("Raw CAM Activation")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], shrink=0.8, label='Activation')

    plt.suptitle("Grad-CAM Explainability", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════
#  Final Report
# ═══════════════════════════════════════════════════════════

def show_final_report(original_img, visualization, severity_label, score,
                      diagnosis, confidence, grid_scores=None):
    """Display the complete analysis report."""
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Original X-Ray
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title(f"Input X-Ray\n{diagnosis}", fontsize=11)
    ax1.axis('off')

    # Panel 2: Grad-CAM Heatmap
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(visualization)
    ax2.set_title("Grad-CAM Heatmap\n(Red = Infection Focus)", fontsize=11)
    ax2.axis('off')

    # Panel 3: Severity Gauge
    ax3 = fig.add_subplot(1, 4, 3)
    colors = ['green', 'orange', 'red']
    thresholds = [0.15, 0.35, 1.0]
    labels = ['Mild', 'Moderate', 'Severe']
    bottom = 0
    for c, t, l in zip(colors, thresholds, labels):
        height = t - bottom
        ax3.bar(['Severity'], [height], bottom=bottom, color=c, alpha=0.3,
                edgecolor='none')
        ax3.text(0, bottom + height / 2, l, ha='center', va='center',
                 fontsize=8, color=c, fontweight='bold')
        bottom = t
    # Actual score marker
    ax3.axhline(y=score, color='black', linewidth=2.5, linestyle='--')
    ax3.text(0.52, score, f" ← {score:.2f}", ha='left', va='center',
             fontsize=10, fontweight='bold',
             transform=ax3.get_yaxis_transform())
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Opacity Ratio")
    ax3.set_title(f"Severity: {severity_label}", fontsize=11, fontweight='bold')

    # Panel 4: Summary Text
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.axis('off')
    report_text = (
        f"{'═' * 28}\n"
        f"  DIAGNOSTIC REPORT\n"
        f"{'═' * 28}\n\n"
        f"  Diagnosis:  {diagnosis}\n"
        f"  Confidence: {confidence:.1%}\n"
        f"  Severity:   {severity_label}\n"
        f"  Opacity:    {score:.4f}\n"
    )
    if grid_scores is not None:
        max_idx = np.unravel_index(np.argmax(grid_scores), grid_scores.shape)
        report_text += (
            f"\n  Grid Peak:  R{max_idx[0]+1}-C{max_idx[1]+1}"
            f" ({grid_scores[max_idx]:.0%})\n"
            f"  Mean Grid:  {grid_scores.mean():.1%}\n"
        )
    report_text += (
        f"\n{'═' * 28}\n"
        f"  ⚠️ Research prototype.\n"
        f"  Not for clinical use.\n"
        f"{'═' * 28}"
    )
    ax4.text(0.05, 0.95, report_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))

    plt.suptitle("🫁 Pneumonia Severity Analysis — Final Report",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Console output
    print("\n✅ ANALYSIS COMPLETE")
    print(f"   Diagnosis:  {diagnosis}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Severity:   {severity_label}")
    print(f"   Opacity:    {score:.4f}")
    print("   ⚠️ DISCLAIMER: Research prototype — not for clinical diagnosis.")
