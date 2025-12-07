#!/usr/bin/env python3
"""
Evaluation script for VANS.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint ./outputs/checkpoints/best_model.pt
"""

import os
import sys
import argparse
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_paths, get_training_config, get_model_config, CONFIG_SHORT, SEED
from src.utils.device import setup_device
from src.data.dataset import load_features, create_dataloaders
from src.models.vans import create_model
from src.training.evaluation import evaluate, confusion_analysis, print_test_results, print_error_analysis
from src.utils.visualization import plot_config_breakdown, plot_paper_figure


def main():
    parser = argparse.ArgumentParser(description='Evaluate VANS model')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    args = parser.parse_args()

    # Get config
    paths = get_paths(output_dir=args.output_dir)
    train_config = get_training_config()
    model_config = get_model_config()

    checkpoint_path = args.checkpoint or os.path.join(paths['checkpoint_dir'], 'best_model.pt')

    print("=" * 60)
    print("VANS Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results directory: {paths['results_dir']}")
    print("=" * 60)

    # Setup device
    device = setup_device(SEED)

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("Please train a model first or specify a valid checkpoint path.")
        sys.exit(1)

    # Load features
    features = load_features(paths['features_dir'])

    # Create dataloaders (only need test loader)
    _, _, test_loader = create_dataloaders(
        features,
        batch_size=train_config['batch_size']
    )

    # Create model
    model = create_model(model_config, device)

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"Validation accuracy at save: {checkpoint['val_acc']:.2%}")

    # Evaluate on test set
    test_acc, test_config_acc = evaluate(model, test_loader, device)

    # Print results
    print_test_results(test_acc, test_config_acc, CONFIG_SHORT)

    # Confusion analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    confusion_data, errors = confusion_analysis(model, test_loader, device)
    print_error_analysis(errors)

    # Prediction distribution
    print("\nPrediction Distribution Analysis:")
    for cfg in sorted(confusion_data.keys()):
        import numpy as np
        data = confusion_data[cfg]
        acc = data['correct'] / data['total'] if data['total'] > 0 else 0
        pred_dist = data['pred_dist'] / data['total'] if data['total'] > 0 else data['pred_dist']
        short_name = CONFIG_SHORT.get(cfg, cfg[:15])

        print(f"\n{short_name}:")
        print(f"  Accuracy: {acc:.2%}")
        print(f"  Prediction distribution: {np.round(pred_dist * 100, 1)}")

        entropy = -np.sum(pred_dist * np.log(pred_dist + 1e-10))
        max_entropy = np.log(8)
        print(f"  Prediction entropy: {entropy:.2f} / {max_entropy:.2f}")

    # Save plots
    os.makedirs(paths['results_dir'], exist_ok=True)

    plot_config_breakdown(
        test_acc, test_config_acc, CONFIG_SHORT,
        save_path=os.path.join(paths['results_dir'], 'config_breakdown.png')
    )

    # Load history for paper figure if available
    if 'history' in checkpoint:
        plot_paper_figure(
            checkpoint['history'], test_acc, test_config_acc,
            checkpoint.get('best_val_acc', checkpoint['val_acc']),
            CONFIG_SHORT,
            save_path=os.path.join(paths['results_dir'], 'paper_figure.png')
        )

    print(f"\n[OK] All results saved to: {paths['results_dir']}/")


if __name__ == '__main__':
    main()
