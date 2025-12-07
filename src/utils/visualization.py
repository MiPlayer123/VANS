"""Visualization utilities for VANS."""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_training_curves(history, best_val_acc, save_path=None):
    """
    Plot training loss and accuracy curves.

    Args:
        history: Dict with 'train_loss', 'train_acc', 'val_acc' lists
        best_val_acc: Best validation accuracy achieved
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    epochs = range(1, len(history['train_acc']) + 1)
    axes[1].plot(epochs, history['train_acc'], label='Train Acc', color='blue')
    axes[1].plot(epochs, history['val_acc'], label='Val Acc', color='orange')
    axes[1].axhline(y=best_val_acc, color='red', linestyle='--', alpha=0.7,
                    label=f'Best: {best_val_acc:.2%}')
    axes[1].axhline(y=0.125, color='gray', linestyle=':', alpha=0.5, label='Random: 12.5%')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Training curves saved to {save_path}")

    plt.show()
    plt.close()


def plot_config_breakdown(test_acc, config_acc, config_short=None, save_path=None):
    """
    Plot per-configuration accuracy breakdown.

    Args:
        test_acc: Overall test accuracy
        config_acc: Dict mapping config name to accuracy
        config_short: Dict mapping full config names to short names
        save_path: Path to save the figure (optional)
    """
    config_short = config_short or {}

    # Create results dataframe
    results_df = pd.DataFrame([
        {'Configuration': config_short.get(cfg, cfg[:15]), 'Accuracy': acc}
        for cfg, acc in config_acc.items()
    ])
    results_df = results_df.sort_values('Accuracy', ascending=True)

    # Bar plot
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(results_df))
    bars = plt.barh(results_df['Configuration'], results_df['Accuracy'], color=colors)

    plt.xlabel('Accuracy', fontsize=12)
    plt.title(f'VANS Performance by Configuration\n(Overall Test Accuracy: {test_acc:.2%})', fontsize=14)
    plt.axvline(x=test_acc, color='red', linestyle='--', linewidth=2, label=f'Overall: {test_acc:.2%}')
    plt.axvline(x=0.125, color='gray', linestyle=':', linewidth=2, label='Random: 12.5%')
    plt.legend(loc='lower right')
    plt.xlim(0, 1)

    # Add value labels
    for bar, acc in zip(bars, results_df['Accuracy']):
        plt.text(acc + 0.02, bar.get_y() + bar.get_height()/2, f'{acc:.1%}',
                 va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Configuration breakdown saved to {save_path}")

    plt.show()
    plt.close()


def plot_paper_figure(history, test_acc, config_acc, best_val_acc, config_short=None, save_path=None):
    """
    Create summary figure suitable for paper/presentation.

    Args:
        history: Training history dict
        test_acc: Overall test accuracy
        config_acc: Dict mapping config name to accuracy
        best_val_acc: Best validation accuracy
        config_short: Dict mapping full config names to short names
        save_path: Path to save the figure (optional)
    """
    config_short = config_short or {}

    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Architecture diagram (text placeholder)
    ax1 = fig.add_subplot(131)
    ax1.text(0.5, 0.7, 'VANS Architecture', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.5, 'DINOv2-L (frozen)\n|\nContext Encoder (4-layer Transformer)\n|\nRule Reasoning (Cross-Attention)\n|\nCandidate Scoring',
             ha='center', va='center', fontsize=10, family='monospace')
    ax1.set_title('(a) Model Architecture', fontsize=12)
    ax1.axis('off')
    ax1.set_facecolor('#f0f0f0')

    # Plot 2: Training curves
    ax2 = fig.add_subplot(132)
    epochs = range(1, len(history['train_acc']) + 1)
    ax2.plot(epochs, history['train_acc'], label='Train', color='blue', linewidth=2)
    ax2.plot(epochs, history['val_acc'], label='Validation', color='orange', linewidth=2)
    ax2.axhline(y=0.125, color='gray', linestyle=':', alpha=0.7, label='Random')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('(b) Training Progress', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Plot 3: Per-config performance
    ax3 = fig.add_subplot(133)
    configs = sorted(config_acc.keys())
    accs = [config_acc[c] for c in configs]
    short_names = [config_short.get(c, c[:10]).replace('_', '\n') for c in configs]

    bars = ax3.bar(range(len(configs)), accs, color='steelblue', edgecolor='navy', linewidth=1)
    ax3.axhline(y=test_acc, color='red', linestyle='--', linewidth=2, label=f'Overall: {test_acc:.1%}')
    ax3.axhline(y=0.125, color='gray', linestyle=':', linewidth=1.5, label='Random: 12.5%')

    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('(c) Per-Configuration Results', fontsize=12)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.0%}',
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Paper figure saved to {save_path}")

    plt.show()
    plt.close()

    print(f"\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\nTest Accuracy: {test_acc:.2%}")
    print(f"Best Validation Accuracy: {best_val_acc:.2%}")
