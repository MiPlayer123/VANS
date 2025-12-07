"""Device detection and GPU optimizations."""

import os

# CRITICAL: Set MPS fallback BEFORE importing torch
# This allows unsupported ops (like bicubic upsample in DINOv2) to fall back to CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import gc
import random
import numpy as np
import torch


def get_device():
    """
    Get the best available device.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def setup_device(seed=42):
    """
    Set up device with optimizations and set random seeds.

    Args:
        seed: Random seed for reproducibility

    Returns:
        torch.device
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()

    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        # GPU optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[OK] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("     GPU optimizations enabled (TF32, cuDNN benchmark)")

    elif device.type == 'mps':
        # Enable MPS fallback for unsupported ops (required for DINOv2)
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("[OK] Using Apple Silicon GPU (MPS)")
        print("     MPS fallback enabled for unsupported ops")

    else:
        print("[WARN] No GPU detected - training will be slow!")

    gc.collect()

    return device


def print_device_info():
    """Print detailed device information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
