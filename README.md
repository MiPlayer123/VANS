# VANS: Visual Analogy Network Solver

A deep learning system for solving Raven's Progressive Matrices (RPM) - visual analogy puzzles used in IQ tests. VANS uses a frozen DINOv2-L backbone for feature extraction combined with a transformer-based architecture for reasoning.

## Features

- **DINOv2-L Backbone**: Frozen pre-trained vision transformer for robust visual features
- **Transformer Context Encoder**: 4-layer encoder with row/column positional embeddings
- **Cross-Attention Reasoning**: Candidates attend to context patterns for scoring
- **Built-in Data Generation**: Generate custom RAVEN datasets using And-Or Tree grammar
- **Multi-Dataset Support**: Trained on the bias-removed I-RAVEN dataset but supports RAVEN generation
- **Multi-Platform Support**: CUDA, Apple Silicon (MPS), and CPU
- **Modular Design**: Separate scripts for extraction, training, and evaluation

## Architecture

```
RAVEN Problem (8 context panels + 8 answer candidates)
                    │
                    ▼
    ┌───────────────────────────────┐
    │  DINOv2-L (frozen)            │  ← 1024-dim features per panel
    │  Pre-extracted & cached       │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Context Encoder              │  ← 4-layer Transformer
    │  Row/Col positional embeddings│     [8 × 1024] → [512]
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Rule Reasoning Module        │  ← Cross-attention mechanism
    │  Score each candidate         │     Candidates attend to context
    └───────────────────────────────┘
                    │
                    ▼
            Softmax → Answer (0-7)
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/VANS.git
cd VANS
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 2.0+

## Quick Start

### Option 1: Use Existing RAVEN Data

1. Download [I-RAVEN](https://github.com/husheng12345/SRAN) or [RAVEN](https://github.com/WellyZhang/RAVEN) dataset
2. Edit `config.py`:
   ```python
   DATA_DIR = "/path/to/RAVEN-10000"
   OUTPUT_DIR = "./outputs"
   ```
3. Run:
   ```bash
   python main.py
   ```

### Option 2: Generate Your Own Data

```bash
# Generate 1000 samples per configuration (7000 total)
python main.py --generate-data --num-generate 1000

# Or use the standalone script
python scripts/generate_data.py --num-samples 1000 --save-dir ./my_data
```

### Test Mode

Quick validation run (10 samples, 1 epoch):
```bash
python main.py --test-mode
```

## Usage

### Full Pipeline

```bash
# Default settings from config.py
python main.py

# Custom paths
python main.py --data-dir /path/to/data --output-dir ./experiment

# Generate data, then train
python main.py --generate-data --num-generate 2000

# Skip feature extraction (if already done)
python main.py --skip-extraction

# Resume from checkpoint
python main.py --resume ./outputs/checkpoints/epoch_050.pt

# Evaluation only
python main.py --eval-only --checkpoint ./outputs/checkpoints/best_model.pt
```

### Individual Scripts

```bash
# Feature extraction only
python scripts/extract_features.py --num-samples 5000

# Training only
python scripts/train.py --epochs 100 --lr 1e-4

# Evaluation only
python scripts/evaluate.py --checkpoint ./outputs/checkpoints/best_model.pt

# Data generation only
python scripts/generate_data.py --num-samples 1000 --save-dir ./data
```

## Project Structure

```
VANS/
├── config.py                 # Central configuration (edit this!)
├── main.py                   # Full pipeline entry point
├── requirements.txt
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # CachedFeatureDataset, dataloaders
│   │   └── preprocessing.py  # PIL-based image preprocessing
│   │
│   ├── datagen/              # RAVEN data generation module
│   │   ├── generator.py      # High-level generation API
│   │   ├── AoT.py            # And-Or Tree structures
│   │   ├── Rule.py           # Rule types (Constant, Progression, etc.)
│   │   ├── rendering.py      # Panel rendering
│   │   ├── sampling.py       # Answer candidate sampling
│   │   └── ...               # Additional generation utilities
│   │
│   ├── features/
│   │   └── extractor.py      # DINOv2 feature extraction with caching
│   │
│   ├── models/
│   │   ├── vans.py           # Main VANS model
│   │   ├── context_encoder.py    # Transformer encoder
│   │   ├── rule_reasoning.py     # Cross-attention module
│   │   └── rule_predictor.py     # Final prediction head
│   │
│   ├── training/
│   │   ├── trainer.py        # Training loop with early stopping
│   │   ├── losses.py         # CE + margin loss
│   │   └── evaluation.py     # Metrics and analysis
│   │
│   └── utils/
│       ├── device.py         # CUDA/MPS/CPU detection
│       └── visualization.py  # Training curves, result plots
│
├── scripts/
│   ├── extract_features.py   # Standalone feature extraction
│   ├── train.py              # Standalone training
│   ├── evaluate.py           # Standalone evaluation
│   └── generate_data.py      # Standalone data generation
│
├── notebooks/
│   └── VANS_Final.ipynb      # Original development notebook
│
└── outputs/                  # Created automatically
    ├── features/             # Cached DINOv2 features (.pt files)
    ├── checkpoints/          # Model checkpoints
    └── results/              # Plots and metrics
```

## Configuration

All settings are in `config.py`. Edit this file to change defaults.

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `DATA_DIR` | `/path/to/RAVEN` | Path to RAVEN dataset |
| `OUTPUT_DIR` | `./outputs` | Output directory |
| `USE_GENERATED_DATA` | `False` | Use generated data instead of existing |
| `NUM_SAMPLES_PER_CONFIG` | `5000` | Samples per configuration (50/2000/5000/10000/None) |
| `BATCH_SIZE` | `64` | Training batch size |
| `MAX_EPOCHS` | `100` | Maximum training epochs |
| `PATIENCE` | `15` | Early stopping patience |
| `LEARNING_RATE` | `1e-4` | Initial learning rate |

### Model Architecture

| Setting | Default | Description |
|---------|---------|-------------|
| `FEATURE_DIM` | `1024` | DINOv2-L output dimension |
| `HIDDEN_DIM` | `512` | Internal model dimension |
| `NUM_HEADS` | `8` | Transformer attention heads |
| `NUM_LAYERS` | `4` | Transformer encoder layers |
| `DROPOUT` | `0.1` | Dropout rate |

### Environment

```python
USE_COLAB = False      # Set True for Google Colab
TEST_MODE = False      # Set True for quick testing
```

## Data Generation

VANS includes a built-in RAVEN dataset generator using And-Or Tree (AoT) grammar.

### Configurations

| Config | Description |
|--------|-------------|
| `center_single` | Single shape in center |
| `distribute_four` | 2×2 grid of shapes |
| `distribute_nine` | 3×3 grid of shapes |
| `left_right` | Left and right panels |
| `up_down` | Upper and lower panels |
| `in_out_center` | Nested shapes (inner/outer) |
| `in_out_grid` | Grid with nested shapes |

### Rule Types

- **Constant**: Attribute remains unchanged across rows
- **Progression**: Attribute changes by fixed increment
- **Arithmetic**: Add/subtract operations on attributes
- **Distribute_Three**: Three values distributed across row

### Generated Data Format

Each sample is saved as an NPZ file containing:
- `image`: 16 panels (8 context + 8 candidates), 160×160 grayscale
- `target`: Correct answer index (0-7)
- `meta_matrix`: Attribute metadata
- `meta_structure`: Structural metadata

## Google Colab

```python
# In config.py
USE_COLAB = True
COLAB_DATA_DIR = "/content/drive/MyDrive/VANS/data/I-RAVEN"
COLAB_OUTPUT_DIR = "/content/drive/MyDrive/VANS"
```

## Results

Training outputs are saved to `outputs/results/`:
- `training_curves.png` - Loss and accuracy over epochs
- `config_breakdown.png` - Per-configuration accuracy
- `paper_figure.png` - Summary figure

Target accuracy: **85-88%** on I-RAVEN test set.

## Dataset Information

### RAVEN / I-RAVEN

- 7 figure configurations
- 10,000 samples per configuration
- 70,000 total samples (60% train, 20% val, 20% test)
- 160×160 grayscale images

**Sources**:
- [RAVEN](https://github.com/WellyZhang/RAVEN) - Original dataset
- [I-RAVEN](https://github.com/husheng12345/SRAN) - Bias-corrected version

## License

MIT License

## Team

**Deep Learning for Computer Vision - Final Project**

- Mikul Saravanan
- Alice Lin
- Angel Wu

## Acknowledgments

- DINOv2 by Meta AI Research
- RAVEN dataset by Zhang et al.
- I-RAVEN by Hu et al.
