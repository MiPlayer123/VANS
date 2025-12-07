#!/usr/bin/env python3
"""
Training script for VANS.

Usage:
    python scripts/train.py
    python scripts/train.py --output-dir ./my_experiment
    python scripts/train.py --resume ./outputs/checkpoints/best_model.pt
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_paths, get_training_config, get_model_config, ensure_dirs, SEED
from src.utils.device import setup_device
from src.data.dataset import load_features, create_dataloaders
from src.models.vans import create_model
from src.training.trainer import Trainer
from src.utils.visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description='Train VANS model')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Max epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    args = parser.parse_args()

    # Get config
    paths = get_paths(output_dir=args.output_dir)
    train_config = get_training_config()
    model_config = get_model_config()

    # Apply CLI overrides
    if args.batch_size:
        train_config['batch_size'] = args.batch_size
    if args.epochs:
        train_config['max_epochs'] = args.epochs
    if args.lr:
        train_config['lr'] = args.lr

    print("=" * 60)
    print("VANS Training")
    print("=" * 60)
    print(f"Features directory:  {paths['features_dir']}")
    print(f"Checkpoint directory: {paths['checkpoint_dir']}")
    print(f"Results directory:    {paths['results_dir']}")
    print(f"Batch size: {train_config['batch_size']}")
    print(f"Max epochs: {train_config['max_epochs']}")
    print(f"Learning rate: {train_config['lr']}")
    print("=" * 60)

    # Setup device
    device = setup_device(SEED)

    # Create directories
    ensure_dirs(paths)

    # Load features
    features = load_features(paths['features_dir'])

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        features,
        batch_size=train_config['batch_size']
    )

    # Create model
    model = create_model(model_config, device)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        paths=paths,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    history, best_val_acc = trainer.train()

    # Plot training curves
    plot_training_curves(
        history, best_val_acc,
        save_path=os.path.join(paths['results_dir'], 'training_curves.png')
    )

    print(f"\n[OK] Training complete!")
    print(f"Best model saved to: {paths['checkpoint_dir']}/best_model.pt")


if __name__ == '__main__':
    main()
