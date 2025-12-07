"""DINOv2 feature extraction.

Pre-extracts features from RAVEN images for efficient training.
Features are cached to disk to avoid re-extraction.
Supports incremental extraction (skips already-processed files).
"""

import os
import glob
import numpy as np
import torch
from tqdm import tqdm

from ..data.preprocessing import preprocess_panels


def load_backbone(device):
    """
    Load DINOv2-L backbone (frozen).

    Args:
        device: torch.device to load model onto

    Returns:
        DINOv2-L model in eval mode with frozen parameters
    """
    print("Loading DINOv2-L (this may take a minute)...")

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    backbone = backbone.eval().to(device)

    # Freeze all parameters
    for param in backbone.parameters():
        param.requires_grad = False

    num_params = sum(p.numel() for p in backbone.parameters())
    print(f"[OK] DINOv2-L loaded: {num_params/1e6:.1f}M params")
    print(f"     Output dimension: 1024")

    return backbone


def extract_features_batch(backbone, panels, device, batch_size=16):
    """
    Extract DINOv2 features for a batch of preprocessed panels.

    Args:
        backbone: DINOv2 model
        panels: torch.Tensor of shape [N, 3, 224, 224] (on CPU)
        device: torch.device for inference
        batch_size: Batch size for feature extraction

    Returns:
        torch.Tensor of shape [N, 1024] (on CPU)
    """
    features = []
    for i in range(0, len(panels), batch_size):
        batch = panels[i:i+batch_size].to(device)
        with torch.no_grad():
            feat = backbone(batch)
        features.append(feat.cpu())
    return torch.cat(features, dim=0)


def extract_all_features(data_dir, features_dir, device, num_samples_per_config=None):
    """
    Extract features from all RAVEN samples and cache to disk.

    Features are stored in format compatible with CachedFeatureDataset:
    - 'context': [8, 1024] - first 8 panels
    - 'candidates': [8, 1024] - last 8 panels
    - 'target': int
    - 'config': str

    Saves incrementally after each configuration to prevent data loss.
    Skips already-processed files for efficient resumption.

    Args:
        data_dir: Path to RAVEN dataset root
        features_dir: Path to save extracted features
        device: torch.device for inference
        num_samples_per_config: Limit samples per config (None = all)

    Returns:
        dict: All extracted features
    """
    features_file = os.path.join(features_dir, 'all_features.pt')

    # Load existing features if any
    all_features = {}
    if os.path.exists(features_file):
        print(f"Loading existing features from {features_file}...")
        all_features = torch.load(features_file, weights_only=False)
        print(f"[OK] Loaded {len(all_features)} existing samples")

    # Auto-detect configurations from data_dir
    configs = sorted([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d))])

    # Count existing samples per config
    existing_by_config = {cfg: set() for cfg in configs}
    for key, data in all_features.items():
        cfg = data.get('config', None)
        if cfg and cfg in existing_by_config:
            # Key is just the filename (e.g., "RAVEN_0_train.npz")
            existing_by_config[cfg].add(key)

    print("\nExisting samples per config:")
    for cfg in configs:
        count = len(existing_by_config[cfg])
        target = num_samples_per_config or 10000
        status = "[OK]" if count >= target * 0.9 else "[  ]"
        print(f"  {status} {cfg[:20]:20} {count:5}/{target}")

    # Check if we have enough samples already
    total_existing = len(all_features)
    total_expected = len(configs) * (num_samples_per_config or 10000)

    if total_existing >= total_expected * 0.9:
        print(f"\n[OK] Features already cached ({total_existing} samples)")
        train_count = sum(1 for k in all_features.keys() if 'train' in k)
        val_count = sum(1 for k in all_features.keys() if 'val' in k)
        test_count = sum(1 for k in all_features.keys() if 'test' in k)
        print(f"     Train: {train_count}, Val: {val_count}, Test: {test_count}")
        return all_features

    # Load backbone (only if we need to extract)
    backbone = load_backbone(device)

    print("\nExtracting features...")
    print(f"Using {num_samples_per_config or 'ALL'} samples per config")
    print(f"Device: {device}")
    print(f"Found {len(configs)} configurations")

    os.makedirs(features_dir, exist_ok=True)

    for config in configs:
        config_dir = os.path.join(data_dir, config)
        all_files = sorted(glob.glob(os.path.join(config_dir, '*.npz')))

        # Limit files if num_samples_per_config is set
        if num_samples_per_config:
            all_files = all_files[:num_samples_per_config]

        # Filter out already processed files
        existing_keys = existing_by_config.get(config, set())
        files_to_process = [f for f in all_files
                          if os.path.basename(f) not in existing_keys]

        cached = len(all_files) - len(files_to_process)
        print(f"\n{config}: {len(all_files)} total, {cached} cached, {len(files_to_process)} to extract")

        if not files_to_process:
            print("  [SKIP] All files already cached")
            continue

        for filepath in tqdm(files_to_process, desc=config[:15]):
            try:
                data = np.load(filepath)
                images = data['image']  # [16, 160, 160]
                target = int(data['target'])

                # Preprocess using PIL (CPU), then extract on GPU
                panels = preprocess_panels(images)
                features = extract_features_batch(backbone, panels, device)

                # Store with context/candidates split
                key = os.path.basename(filepath)
                all_features[key] = {
                    'context': features[:8],      # [8, 1024]
                    'candidates': features[8:],   # [8, 1024]
                    'target': target,
                    'config': config
                }
            except Exception as e:
                print(f"\nError: {filepath}: {e}")
                continue

        # Save checkpoint after each config
        torch.save(all_features, features_file)
        print(f"  Saved: {len(all_features)} samples total")

    # Final stats
    print(f"\n[OK] Feature extraction complete!")
    print(f"     Total: {len(all_features)} samples")
    train_count = sum(1 for k in all_features.keys() if 'train' in k)
    val_count = sum(1 for k in all_features.keys() if 'val' in k)
    test_count = sum(1 for k in all_features.keys() if 'test' in k)
    print(f"     Train: {train_count}, Val: {val_count}, Test: {test_count}")

    return all_features


def convert_legacy_features(features_dict):
    """
    Convert features from legacy format (single 'features' tensor) to
    new format (separate 'context' and 'candidates').

    Args:
        features_dict: Dictionary that may contain legacy format

    Returns:
        dict: Converted features in new format
    """
    converted = {}
    for key, data in features_dict.items():
        if 'features' in data and 'context' not in data:
            # Legacy format: 'features' is [16, 1024]
            features = data['features']
            converted[key] = {
                'context': features[:8],
                'candidates': features[8:],
                'target': data['target'],
                'config': data['config']
            }
        else:
            # Already in new format
            converted[key] = data
    return converted


def verify_features(features_dir):
    """
    Verify cached features and print statistics.

    Args:
        features_dir: Path to features directory

    Returns:
        dict: Loaded features if valid, None otherwise
    """
    features_file = os.path.join(features_dir, 'all_features.pt')

    if not os.path.exists(features_file):
        print(f"[ERROR] Features file not found: {features_file}")
        return None

    print("Loading cached features...")
    features = torch.load(features_file, weights_only=False)

    # Convert legacy format if needed
    sample_key = list(features.keys())[0]
    if 'features' in features[sample_key] and 'context' not in features[sample_key]:
        print("[INFO] Converting legacy feature format...")
        features = convert_legacy_features(features)
        torch.save(features, features_file)
        print("[OK] Converted and saved")

    print(f"[OK] Total samples: {len(features)}")

    # Check one sample
    sample = features[sample_key]
    print(f"\nSample: {sample_key}")
    print(f"  Context shape: {sample['context'].shape}")
    print(f"  Candidates shape: {sample['candidates'].shape}")
    print(f"  Target: {sample['target']}")
    print(f"  Config: {sample['config']}")

    # Count by split
    splits = {'train': 0, 'val': 0, 'test': 0}
    configs_count = {}

    for key in features.keys():
        for split in splits:
            if split in key:
                splits[split] += 1
                break
        cfg = features[key]['config']
        configs_count[cfg] = configs_count.get(cfg, 0) + 1

    print(f"\nSplit counts: {splits}")
    print(f"Config counts: {configs_count}")

    return features
