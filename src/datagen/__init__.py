"""
RAVEN Dataset Generator

This module contains code for generating RAVEN Progressive Matrices datasets.
Based on the original RAVEN implementation by Zhang et al. (CVPR 2019).

Usage:
    from src.datagen import generate_dataset

    generate_dataset(
        num_samples=1000,
        save_dir='./generated_data',
        seed=42
    )
"""

from .generator import generate_dataset, generate_single_config

__all__ = ['generate_dataset', 'generate_single_config']
