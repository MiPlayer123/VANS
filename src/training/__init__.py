"""Training utilities for VANS."""

from .losses import primary_loss, contrastive_margin_loss, compute_loss
from .trainer import Trainer, train_epoch
from .evaluation import evaluate, confusion_analysis
