#!/usr/bin/env python3
"""
RAVEN Dataset Generator - Standalone Script

Generates RAVEN Progressive Matrices dataset using the VANS datagen module.

Usage:
    # Generate 1000 samples per config (7000 total)
    python scripts/generate_data.py --num-samples 1000 --save-dir ./generated_data

    # Quick test (10 samples per config)
    python scripts/generate_data.py --num-samples 10 --save-dir ./test_data

    # Generate specific configurations only
    python scripts/generate_data.py --num-samples 100 --configs center_single distribute_four
"""

import os
import sys
import argparse

# Ensure we can import from the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datagen import generate_dataset, generate_single_config
from src.datagen.generator import ALL_CONFIGS


def main():
    parser = argparse.ArgumentParser(
        description='Generate RAVEN Progressive Matrices dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_data.py --num-samples 1000
  python scripts/generate_data.py --num-samples 100 --configs center_single
  python scripts/generate_data.py --num-samples 10 --save-dir ./test_data
        """
    )

    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples per configuration (default: 1000)')
    parser.add_argument('--save-dir', type=str, default='./generated_data',
                        help='Output directory (default: ./generated_data)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--configs', nargs='+', default=None,
                        help=f'Configurations to generate (default: all). Available: {list(ALL_CONFIGS.keys())}')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    parser.add_argument('--save-xml', action='store_true',
                        help='Save XML annotations (default: False)')

    args = parser.parse_args()

    # Validate configs
    if args.configs:
        for config in args.configs:
            if config not in ALL_CONFIGS:
                print(f"Error: Unknown config '{config}'")
                print(f"Available: {list(ALL_CONFIGS.keys())}")
                sys.exit(1)

    print("=" * 60)
    print("RAVEN Dataset Generator")
    print("=" * 60)

    results = generate_dataset(
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        seed=args.seed,
        configs=args.configs,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        save_xml=args.save_xml
    )

    print("\nGeneration complete!")
    print(f"Dataset saved to: {os.path.abspath(args.save_dir)}")


if __name__ == '__main__':
    main()
