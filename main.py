#!/usr/bin/env python3
"""
VANS: Visual Analogy Network Solver - Main Entry Point

Runs the full pipeline: feature extraction -> training -> evaluation

Usage:
    # Run with defaults from config.py
    python main.py

    # Override data/output directories
    python main.py --data-dir /path/to/RAVEN --output-dir ./my_experiment

    # Quick test mode
    python main.py --test-mode

    # Skip feature extraction (if already done)
    python main.py --skip-extraction

    # Resume training from checkpoint
    python main.py --resume ./outputs/checkpoints/best_model.pt

    # Evaluation only
    python main.py --eval-only --checkpoint ./outputs/checkpoints/best_model.pt
"""

import os
# CRITICAL: Set MPS fallback BEFORE any torch import
# This allows unsupported ops (like bicubic upsample in DINOv2) to fall back to CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import argparse

# Ensure we can import from the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import get_paths, get_training_config, get_model_config, ensure_dirs, print_config, CONFIG_SHORT


def main():
    parser = argparse.ArgumentParser(
        description='VANS: Visual Analogy Network Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run with defaults
  python main.py --data-dir /path/to/data     # Custom data path
  python main.py --test-mode                  # Quick test (1 epoch)
  python main.py --skip-extraction            # Skip feature extraction
  python main.py --eval-only                  # Evaluation only
        """
    )

    # Path arguments
    parser.add_argument('--data-dir', type=str, help='Path to RAVEN dataset')
    parser.add_argument('--output-dir', type=str, help='Output directory for features, checkpoints, results')

    # Mode arguments
    parser.add_argument('--test-mode', action='store_true', help='Quick test mode (1 epoch, 50 samples)')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip feature extraction')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')

    # Training arguments
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for evaluation')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Max epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')

    args = parser.parse_args()

    # Apply test mode if specified (modifies config module)
    if args.test_mode:
        config.TEST_MODE = True
        print("[TEST MODE] Running quick test (1 epoch, 50 samples)")

    # Get configuration
    paths = get_paths(data_dir=args.data_dir, output_dir=args.output_dir)
    train_config = get_training_config()
    model_config = get_model_config()

    # Apply CLI overrides
    if args.batch_size:
        train_config['batch_size'] = args.batch_size
    if args.epochs:
        train_config['max_epochs'] = args.epochs
    if args.lr:
        train_config['lr'] = args.lr

    # Print configuration
    print_config()

    # Create output directories
    ensure_dirs(paths)

    # Import here to avoid slow imports if just checking help
    from src.utils.device import setup_device
    from src.features.extractor import extract_all_features, verify_features
    from src.data.dataset import load_features, create_dataloaders
    from src.models.vans import create_model
    from src.training.trainer import Trainer
    from src.training.evaluation import evaluate, confusion_analysis, print_test_results, print_error_analysis
    from src.utils.visualization import plot_training_curves, plot_config_breakdown, plot_paper_figure
    import torch

    # Setup device
    device = setup_device(config.SEED)

    # ========================
    # STEP 1: Feature Extraction
    # ========================
    if not args.skip_extraction and not args.eval_only:
        print("\n" + "=" * 60)
        print("STEP 1: Feature Extraction")
        print("=" * 60)

        features = extract_all_features(
            data_dir=paths['data_dir'],
            features_dir=paths['features_dir'],
            device=device,
            num_samples_per_config=train_config['num_samples_per_config']
        )
    else:
        print("\n[SKIP] Feature extraction skipped")
        features = None

    # ========================
    # STEP 2: Training
    # ========================
    if not args.eval_only:
        print("\n" + "=" * 60)
        print("STEP 2: Training")
        print("=" * 60)

        # Load features if not already loaded
        if features is None:
            features = load_features(paths['features_dir'])

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
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

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # Train
        history, best_val_acc = trainer.train()

        # Save training curves
        plot_training_curves(
            history, best_val_acc,
            save_path=os.path.join(paths['results_dir'], 'training_curves.png')
        )

    # ========================
    # STEP 3: Evaluation
    # ========================
    print("\n" + "=" * 60)
    print("STEP 3: Evaluation")
    print("=" * 60)

    checkpoint_path = args.checkpoint or os.path.join(paths['checkpoint_dir'], 'best_model.pt')

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("Training may have failed or no checkpoint was saved.")
        return

    # Load features if needed
    if features is None or args.eval_only:
        features = load_features(paths['features_dir'])

    # Create test loader
    if args.eval_only:
        _, _, test_loader = create_dataloaders(
            features,
            batch_size=train_config['batch_size']
        )
        model = create_model(model_config, device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"Validation accuracy at save: {checkpoint['val_acc']:.2%}")

    # Evaluate
    test_acc, test_config_acc = evaluate(model, test_loader, device)

    # Print results
    print_test_results(test_acc, test_config_acc, CONFIG_SHORT)

    # Confusion analysis
    confusion_data, errors = confusion_analysis(model, test_loader, device)
    print_error_analysis(errors)

    # Save plots
    plot_config_breakdown(
        test_acc, test_config_acc, CONFIG_SHORT,
        save_path=os.path.join(paths['results_dir'], 'config_breakdown.png')
    )

    if 'history' in checkpoint:
        plot_paper_figure(
            checkpoint['history'], test_acc, test_config_acc,
            checkpoint.get('best_val_acc', checkpoint['val_acc']),
            CONFIG_SHORT,
            save_path=os.path.join(paths['results_dir'], 'paper_figure.png')
        )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Results saved to: {paths['results_dir']}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
