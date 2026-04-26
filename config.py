"""
Pneumonia Detection System — Configuration
All hyperparameters and paths centralized here.
"""
import torch

# ─── Paths ───────────────────────────────────────────────
DATA_DIR = './Pneumonia Detection Dataset/chest_xray'
MODEL_SAVE_PATH = 'pneumonia_model.pth'

# ─── Image Settings ─────────────────────────────────────
IMAGE_SIZE = 224
GRAYSCALE_MEAN = [0.485]
GRAYSCALE_STD = [0.229]

# ─── Training Hyperparameters ───────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 5
VAL_SPLIT_RATIO = 0.2
NUM_CLASSES = 2
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# ─── Severity Thresholds ────────────────────────────────
OPACITY_LOWER = 150
OPACITY_UPPER = 255
LUNG_THRESHOLD = 50
SEVERITY_MILD_UPPER = 0.15
SEVERITY_MODERATE_UPPER = 0.35

# ─── CLAHE Settings ─────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# ─── Device ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
