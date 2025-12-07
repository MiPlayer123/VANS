"""Training loop and Trainer class for VANS."""

import os
import math
import torch
from tqdm import tqdm

from .losses import compute_loss
from .evaluation import evaluate


def train_epoch(model, loader, optimizer, scaler, device, config):
    """
    Train for one epoch.

    Args:
        model: VANS model
        loader: Training DataLoader
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: torch.device
        config: Training config dict

    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        context = batch['context'].to(device)
        candidates = batch['candidates'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            scores, _ = model(context, candidates)
            loss, loss_dict = compute_loss(
                scores, targets,
                alpha=config['alpha'],
                gamma=config['gamma'],
                margin=config.get('margin', 0.5)
            )

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item()
        preds = scores.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.2%}'})

    return total_loss / len(loader), correct / total


class Trainer:
    """
    Trainer class for VANS with checkpointing and early stopping.
    """

    def __init__(self, model, train_loader, val_loader, config, paths, device):
        """
        Args:
            model: VANS model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training config dict
            paths: Paths dict with checkpoint_dir, results_dir
            device: torch.device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.paths = paths
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # LR Scheduler with warmup + cosine decay
        def lr_lambda(epoch):
            if epoch < config['warmup_epochs']:
                return (epoch + 1) / max(1, config['warmup_epochs'])
            progress = (epoch - config['warmup_epochs']) / max(1, config['max_epochs'] - config['warmup_epochs'])
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        # Training state
        self.best_val_acc = 0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        self.start_epoch = 0

    def train(self):
        """
        Run the full training loop.

        Returns:
            history: Training history dict
            best_val_acc: Best validation accuracy achieved
        """
        print("Starting training...")
        print(f"Max epochs: {self.config['max_epochs']}, Early stopping patience: {self.config['patience']}")
        print()

        for epoch in range(self.start_epoch, self.config['max_epochs']):
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{self.config['max_epochs']} (LR: {current_lr:.6f})")

            # Train
            train_loss, train_acc = train_epoch(
                self.model, self.train_loader, self.optimizer,
                self.scaler, self.device, self.config
            )
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            val_acc, val_config_acc = evaluate(self.model, self.val_loader, self.device)
            self.history['val_acc'].append(val_acc)

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")

            # Per-config results (compact)
            from config import CONFIG_SHORT
            config_str = " | ".join([f"{CONFIG_SHORT.get(k, k[:3])}:{v:.0%}"
                                    for k, v in sorted(val_config_acc.items())])
            print(f"  Configs: {config_str}")

            # Checkpoint best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_acc, val_config_acc, is_best=True)
                print(f"  [OK] New best model saved! ({val_acc:.2%})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['patience']:
                    print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {self.config['patience']} epochs)")
                    break

            # Periodic checkpoint
            if (epoch + 1) % self.config.get('save_every_n_epochs', 10) == 0:
                self._save_checkpoint(epoch, val_acc, val_config_acc, is_best=False)

            self.scheduler.step()
            print()

        # Always save final checkpoint if no best model was saved
        best_path = os.path.join(self.paths['checkpoint_dir'], 'best_model.pt')
        if not os.path.exists(best_path):
            # Save current state as best model (even if accuracy is 0)
            self._save_checkpoint(epoch, val_acc, val_config_acc, is_best=True)
            print("  [OK] Final model saved as best_model.pt")

        print(f"\n" + "="*50)
        print(f"Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2%}")
        print(f"="*50)

        return self.history, self.best_val_acc

    def _save_checkpoint(self, epoch, val_acc, config_acc, is_best=False):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config_acc': config_acc,
            'history': self.history,
            'best_val_acc': self.best_val_acc,
        }

        os.makedirs(self.paths['checkpoint_dir'], exist_ok=True)

        if is_best:
            path = os.path.join(self.paths['checkpoint_dir'], 'best_model.pt')
        else:
            path = os.path.join(self.paths['checkpoint_dir'], f'epoch_{epoch+1:03d}.pt')

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint.get('best_val_acc', checkpoint['val_acc'])
        self.start_epoch = checkpoint['epoch'] + 1

        print(f"Resumed from epoch {self.start_epoch}, best val acc: {self.best_val_acc:.2%}")
