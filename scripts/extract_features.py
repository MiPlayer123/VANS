#!/usr/bin/env python3
"""
Standalone feature extraction script.

Usage:
    python scripts/extract_features.py
    python scripts/extract_features.py --data-dir /path/to/data --output-dir ./outputs
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_paths, get_training_config, SEED
from src.utils.device import setup_device
from src.features.extractor import extract_all_features, verify_features


def main():
    parser = argparse.ArgumentParser(description='Extract DINOv2 features from RAVEN dataset')
    parser.add_argument('--data-dir', type=str, help='Path to RAVEN dataset')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--num-samples', type=int, help='Samples per config (default from config.py)')
    args = parser.parse_args()

    # Get paths and config
    paths = get_paths(data_dir=args.data_dir, output_dir=args.output_dir)
    train_config = get_training_config()

    num_samples = args.num_samples or train_config['num_samples_per_config']

    print("=" * 60)
    print("VANS Feature Extraction")
    print("=" * 60)
    print(f"Data directory:    {paths['data_dir']}")
    print(f"Features directory: {paths['features_dir']}")
    print(f"Samples per config: {num_samples or 'ALL'}")
    print("=" * 60)

    # Setup device
    device = setup_device(SEED)

    # Create output directories
    os.makedirs(paths['features_dir'], exist_ok=True)

    # Extract features
    features = extract_all_features(
        data_dir=paths['data_dir'],
        features_dir=paths['features_dir'],
        device=device,
        num_samples_per_config=num_samples
    )

    # Verify
    print("\nVerifying extracted features...")
    verify_features(paths['features_dir'])

    print("\n[OK] Feature extraction complete!")


if __name__ == '__main__':
    main()
