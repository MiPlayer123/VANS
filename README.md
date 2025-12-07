# VANS: Visual Analogy Network Solver

A deep learning system that solves Raven's Progressive Matrices (RPM) - visual analogy puzzles used in IQ tests.

## Architecture

```
I-RAVEN Problem (8 context + 8 candidates)
              │
              ▼
┌─────────────────────────────┐
│  DINOv2-L (frozen)          │  ← 1024-dim features per image
│  Pre-extract all features   │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Context Encoder            │  ← 4-layer Transformer + positional embeddings
│  [8 x 1024] → [512]         │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Rule Reasoning Module      │  ← Cross-attention: candidates attend to context
│  Score 8 candidates         │
└─────────────────────────────┘
              │
              ▼
        Softmax → Answer
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Paths

Edit `config.py` to set your paths:

```python
# config.py
DATA_DIR = "/path/to/RAVEN-10000"  # Where your RAVEN data lives
OUTPUT_DIR = "./outputs"            # Where to save everything
```

### 3. Run

```bash
# Run full pipeline (extract features → train → evaluate)
python main.py

# Quick test mode (1 epoch, 50 samples)
python main.py --test-mode

# Skip feature extraction (if already done)
python main.py --skip-extraction

# Evaluation only
python main.py --eval-only
```

## Project Structure

```
VANS/
├── config.py                 # ⭐ EDIT THIS - All configuration in one place
├── main.py                   # Run everything: extract → train → evaluate
├── src/
│   ├── data/
│   │   ├── dataset.py        # CachedFeatureDataset
│   │   └── preprocessing.py  # Image preprocessing
│   ├── features/
│   │   └── extractor.py      # DINOv2 feature extraction
│   ├── models/
│   │   ├── context_encoder.py
│   │   ├── rule_reasoning.py
│   │   ├── rule_predictor.py
│   │   └── vans.py           # Main VANS model
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── evaluation.py
│   └── utils/
│       ├── device.py         # GPU/MPS/CPU detection
│       └── visualization.py
├── scripts/
│   ├── extract_features.py   # Standalone extraction
│   ├── train.py              # Training only
│   └── evaluate.py           # Evaluation only
├── notebooks/
│   └── VANS_Final.ipynb      # Original notebook
└── outputs/                  # Created automatically
    ├── features/             # Cached DINOv2 features
    ├── checkpoints/          # Model checkpoints
    └── results/              # Plots and metrics
```

## Configuration

All configuration is in `config.py`. Key settings:

### Paths
```python
DATA_DIR = "/path/to/RAVEN-10000"
OUTPUT_DIR = "./outputs"
```

### Training
```python
NUM_SAMPLES_PER_CONFIG = 5000  # 50 (quick), 2000 (fast), 5000 (good), 10000 (full)
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
```

### Model
```python
HIDDEN_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.1
```

### Test Mode
Set `TEST_MODE = True` in config.py or use `--test-mode` flag for quick testing (1 epoch, 50 samples).

## Command Line Options

```bash
# Full pipeline
python main.py --data-dir /path/to/data --output-dir ./experiment1

# Training only
python scripts/train.py --epochs 50 --lr 0.0001

# Feature extraction only
python scripts/extract_features.py --num-samples 2000

# Evaluation only
python scripts/evaluate.py --checkpoint ./outputs/checkpoints/best_model.pt

# Resume training
python main.py --resume ./outputs/checkpoints/epoch_050.pt
```

## Google Colab

Set `USE_COLAB = True` in `config.py` and update the Colab paths:

```python
USE_COLAB = True
COLAB_DATA_DIR = "/content/drive/MyDrive/VANS/data/I-RAVEN"
COLAB_OUTPUT_DIR = "/content/drive/MyDrive/VANS"
```

## Dataset

Uses the [RAVEN](https://github.com/WellyZhang/RAVEN) dataset (specifically I-RAVEN, the bias-corrected version).

- 7 configurations
- 10,000 samples each (6000 train, 2000 val, 2000 test)
- 70,000 total samples

## Results

Target accuracy: 85-88% on I-RAVEN test set.

Results are saved to `outputs/results/`:
- `training_curves.png` - Training loss and accuracy
- `config_breakdown.png` - Per-configuration accuracy
- `paper_figure.png` - Summary figure for papers

## Team

**Deep Learning for Computer Vision - Final Project**

- Mikul Saravanan
- Alice Lin
- Angel Wu
