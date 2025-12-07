"""Dataset and DataLoader creation for VANS."""

import os
import torch
from torch.utils.data import Dataset, DataLoader


class CachedFeatureDataset(Dataset):
    """
    Dataset using pre-extracted DINOv2 features.

    Loads features from a cached .pt file for efficient training.
    Supports both new format (context/candidates) and legacy format (features).
    """

    def __init__(self, features_dict, split='train'):
        """
        Args:
            features_dict: Dictionary loaded from all_features.pt
            split: 'train', 'val', or 'test'
        """
        self.samples = []
        for key, value in features_dict.items():
            if split in key:
                # Handle both new format (context/candidates) and legacy (features)
                if 'context' in value:
                    context = value['context']
                    candidates = value['candidates']
                elif 'features' in value:
                    # Legacy format: features is [16, 1024]
                    features = value['features']
                    context = features[:8]
                    candidates = features[8:]
                else:
                    print(f"Warning: Unknown format for {key}, skipping")
                    continue

                self.samples.append({
                    'context': context,       # [8, 1024]
                    'candidates': candidates, # [8, 1024]
                    'target': value['target'],
                    'config': value['config'],
                    'key': key
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'context': sample['context'].float(),
            'candidates': sample['candidates'].float(),
            'target': torch.tensor(sample['target'], dtype=torch.long),
            'config': sample['config']
        }


def create_dataloaders(features_dict, batch_size=64, num_workers=0):
    """
    Create train, val, and test DataLoaders from cached features.

    Args:
        features_dict: Dictionary loaded from all_features.pt
        batch_size: Batch size for training
        num_workers: Number of data loading workers (0 for local)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = CachedFeatureDataset(features_dict, split='train')
    val_dataset = CachedFeatureDataset(features_dict, split='val')
    test_dataset = CachedFeatureDataset(features_dict, split='test')

    print(f"[OK] Datasets created:")
    print(f"     Train: {len(train_dataset)} samples")
    print(f"     Val:   {len(val_dataset)} samples")
    print(f"     Test:  {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n[OK] DataLoaders created (batch_size={batch_size}):")
    print(f"     Train batches: {len(train_loader)}")
    print(f"     Val batches:   {len(val_loader)}")
    print(f"     Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


def load_features(features_dir):
    """
    Load cached features from disk.

    Args:
        features_dir: Path to features directory

    Returns:
        dict: Loaded features
    """
    features_file = os.path.join(features_dir, 'all_features.pt')

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")

    print("Loading cached features...")
    features = torch.load(features_file, weights_only=False)
    print(f"[OK] Loaded {len(features)} samples")

    return features
