"""
VANS Configuration
==================
Edit your defaults here. This is the single source of truth for all configuration.

Usage:
    from config import get_paths, get_training_config, get_model_config

    paths = get_paths()
    train_cfg = get_training_config()
"""

import os

# ============================================================
# EDIT YOUR DEFAULTS HERE
# ============================================================

# =========================
# ENVIRONMENT
# =========================
USE_COLAB = False  # Set to True when running on Google Colab
TEST_MODE = False  # Set to True for quick testing (10 samples, 1 epoch)

# =========================
# PATHS - Set your defaults here
# =========================
DATA_DIR = "/Users/mikul/Downloads/RAVEN-10000"  # Where RAVEN data lives
# DATA_DIR = "./data/RAVEN-10000"  # Alternative: relative path
OUTPUT_DIR = "../VANS_output"  # Where to save everything (features, checkpoints, results)
# OUTPUT_DIR = "./outputs"  # Alternative: outputs inside repo

# Colab overrides (used when USE_COLAB=True)
COLAB_DATA_DIR = "/content/drive/MyDrive/VANS/data/I-RAVEN"
COLAB_OUTPUT_DIR = "/content/drive/MyDrive/VANS"

# =========================
# DATA
# =========================
NUM_SAMPLES_PER_CONFIG = 5000  # Samples per configuration. Options: 50, 2000, 5000, 10000, None (all)
SEED = 42

# =========================
# MODEL ARCHITECTURE
# =========================
FEATURE_DIM = 1024   # DINOv2-L output dimension (don't change unless using different backbone)
HIDDEN_DIM = 512     # Internal model dimension
NUM_HEADS = 8        # Transformer attention heads
NUM_LAYERS = 4       # Transformer encoder layers
DROPOUT = 0.1

# =========================
# TRAINING
# =========================
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15        # Early stopping patience
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5

# =========================
# LOSS
# =========================
LOSS_ALPHA = 1.0     # Cross-entropy loss weight
LOSS_GAMMA = 0.1     # Margin loss weight
MARGIN = 0.5         # Contrastive margin

# =========================
# CHECKPOINTING
# =========================
SAVE_EVERY_N_EPOCHS = 10    # Save periodic checkpoints
KEEP_LAST_N_CHECKPOINTS = 3  # Keep last N periodic checkpoints

# =========================
# DISPLAY NAMES
# =========================
CONFIG_SHORT = {
    'center_single': 'center',
    'distribute_four': 'dist_4',
    'distribute_nine': 'dist_9',
    'in_center_single_out_center_single': 'in_out_center',
    'in_distribute_four_out_center_single': 'in_out_grid',
    'left_center_single_right_center_single': 'left_right',
    'up_center_single_down_center_single': 'up_down',
}


# ============================================================
# AUTO-COMPUTED (Don't edit below unless you know what you're doing)
# ============================================================

def get_paths(data_dir=None, output_dir=None):
    """
    Returns resolved paths based on environment.

    Args:
        data_dir: Override DATA_DIR (for CLI usage)
        output_dir: Override OUTPUT_DIR (for CLI usage)

    Returns:
        dict with keys: data_dir, output_dir, features_dir, checkpoint_dir, results_dir
    """
    if USE_COLAB:
        base = output_dir or COLAB_OUTPUT_DIR
        data = data_dir or COLAB_DATA_DIR
    else:
        base = output_dir or OUTPUT_DIR
        data = data_dir or DATA_DIR

    return {
        'data_dir': data,
        'output_dir': base,
        'features_dir': os.path.join(base, 'features'),
        'checkpoint_dir': os.path.join(base, 'checkpoints'),
        'results_dir': os.path.join(base, 'results'),
    }


def get_training_config():
    """
    Returns training config, adjusted for TEST_MODE.

    Returns:
        dict with training hyperparameters
    """
    if TEST_MODE:
        return {
            'batch_size': 4,
            'max_epochs': 1,
            'patience': 1,
            'warmup_epochs': 0,
            'num_samples_per_config': 10,  # Only 10 samples per config = 70 total
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'alpha': LOSS_ALPHA,
            'gamma': LOSS_GAMMA,
            'margin': MARGIN,
        }
    return {
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'warmup_epochs': WARMUP_EPOCHS,
        'num_samples_per_config': NUM_SAMPLES_PER_CONFIG,
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'alpha': LOSS_ALPHA,
        'gamma': LOSS_GAMMA,
        'margin': MARGIN,
    }


def get_model_config():
    """
    Returns model architecture config.

    Returns:
        dict with model hyperparameters
    """
    return {
        'feature_dim': FEATURE_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_heads': NUM_HEADS,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
    }


def ensure_dirs(paths):
    """Create all output directories if they don't exist."""
    for key in ['features_dir', 'checkpoint_dir', 'results_dir']:
        os.makedirs(paths[key], exist_ok=True)


def print_config():
    """Print current configuration."""
    paths = get_paths()
    train_cfg = get_training_config()
    model_cfg = get_model_config()

    print("=" * 60)
    print("VANS CONFIGURATION")
    print("=" * 60)
    print(f"USE_COLAB:              {USE_COLAB}")
    print(f"TEST_MODE:              {TEST_MODE}")
    print(f"SEED:                   {SEED}")
    print()
    print("Paths:")
    for k, v in paths.items():
        print(f"  {k:20} {v}")
    print()
    print("Training:")
    for k, v in train_cfg.items():
        print(f"  {k:20} {v}")
    print()
    print("Model:")
    for k, v in model_cfg.items():
        print(f"  {k:20} {v}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
