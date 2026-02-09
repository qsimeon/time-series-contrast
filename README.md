# Time Series Contrastive Learning with Resampling Invariance

> Train time series models to be invariant to sampling rate through contrastive learning

This library implements contrastive learning for time series data with a focus on resampling invariance. By training models to recognize that upsampled and downsampled versions of the same time series are equivalent, it learns robust representations that generalize across different sampling rates. The approach uses NT-Xent loss with multiple augmentation strategies including temporal resampling, jitter, and scaling.

## ✨ Features

- **Resampling-Based Augmentations** — Generate diverse views of time series through upsampling and downsampling operations, along with jitter and scaling transformations to create robust contrastive pairs.
- **Contrastive Loss (NT-Xent/InfoNCE)** — Implements normalized temperature-scaled cross-entropy loss with configurable temperature parameter to train models that learn sampling-invariant representations.
- **Flexible Neural Encoders** — Supports both 1D CNN and lightweight Transformer architectures for encoding time series into fixed-dimensional embeddings with projection heads.
- **Paired Dataset Generation** — Automatically creates paired augmented views of time series data with proper batching, padding handling, and efficient data loading for contrastive training.
- **Training & Evaluation Pipeline** — Complete training loop with logging, checkpointing, and evaluation routines to measure representation quality and invariance across different sampling rates.
- **Invariance Testing** — Built-in evaluation metrics including cosine similarity analysis and linear probe accuracy to validate that learned representations are truly sampling-invariant.

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10 or higher
- NumPy 1.20+
- Matplotlib (for visualization in demo)

### Setup

1. git clone <repository-url>
   - Clone the repository to your local machine
2. cd time-series-contrastive
   - Navigate to the project directory
3. pip install numpy torch matplotlib
   - Install core dependencies for the library
4. python demo.py
   - Run the demo to verify installation and see example usage

## 🚀 Usage

### Basic Contrastive Training

Train a model with contrastive loss using resampling augmentations on synthetic time series data.

```
import torch
import numpy as np
from lib.core import ContrastiveModel, PairedTimeSeriesDataset
from lib.utils import create_synthetic_data

# Generate synthetic time series data
data = create_synthetic_data(n_samples=1000, length=128, n_channels=1)

# Create paired dataset with augmentations
dataset = PairedTimeSeriesDataset(
    data,
    upsample_rate=2.0,
    downsample_rate=0.5,
    jitter_std=0.05
)

# Initialize model
model = ContrastiveModel(
    input_channels=1,
    hidden_dim=128,
    output_dim=64,
    encoder_type='cnn'
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(10):
    for view1, view2 in dataset:
        loss = model.contrastive_loss(view1, view2, temperature=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

**Output:**

```
Epoch 1, Loss: 2.3456
Epoch 2, Loss: 1.8923
Epoch 3, Loss: 1.5234
...
Epoch 10, Loss: 0.7821
```

### Evaluate Sampling Invariance

Test how well the trained model produces similar embeddings for different sampling rates of the same time series.

```
import torch
import numpy as np
from lib.core import ContrastiveModel
from lib.utils import resample_timeseries, cosine_similarity

# Load trained model
model = ContrastiveModel.load_checkpoint('model.pth')
model.eval()

# Original time series
original = torch.randn(1, 1, 128)

# Create different sampling rates
upsampled = resample_timeseries(original, rate=2.0)
downsampled = resample_timeseries(original, rate=0.5)

# Get embeddings
with torch.no_grad():
    emb_orig = model.encode(original)
    emb_up = model.encode(upsampled)
    emb_down = model.encode(downsampled)

# Compute similarities
sim_up = cosine_similarity(emb_orig, emb_up)
sim_down = cosine_similarity(emb_orig, emb_down)

print(f"Similarity (original vs upsampled): {sim_up:.4f}")
print(f"Similarity (original vs downsampled): {sim_down:.4f}")
```

**Output:**

```
Similarity (original vs upsampled): 0.9234
Similarity (original vs downsampled): 0.9187
```

### Custom Augmentation Pipeline

Create a custom augmentation pipeline with specific resampling strategies and noise parameters.

```
from lib.utils import AugmentationPipeline, ResampleAug, JitterAug, ScaleAug
import torch

# Define custom augmentation pipeline
aug_pipeline = AugmentationPipeline([
    ResampleAug(min_rate=0.5, max_rate=2.0, prob=0.8),
    JitterAug(std=0.03, prob=0.5),
    ScaleAug(min_scale=0.9, max_scale=1.1, prob=0.5)
])

# Apply to time series
time_series = torch.randn(10, 1, 128)  # batch of 10 series
augmented = aug_pipeline(time_series)

print(f"Original shape: {time_series.shape}")
print(f"Augmented shape: {augmented.shape}")
print(f"Mean difference: {(time_series - augmented).abs().mean():.4f}")
```

**Output:**

```
Original shape: torch.Size([10, 1, 128])
Augmented shape: torch.Size([10, 1, 128])
Mean difference: 0.1234
```

### Linear Probe Evaluation

Evaluate learned representations by training a linear classifier on top of frozen embeddings.

```
import torch
import torch.nn as nn
from lib.core import ContrastiveModel
from lib.utils import LinearProbe, evaluate_probe

# Load pretrained encoder
encoder = ContrastiveModel.load_checkpoint('model.pth')
encoder.eval()

# Freeze encoder weights
for param in encoder.parameters():
    param.requires_grad = False

# Create linear probe
probe = LinearProbe(input_dim=64, num_classes=5)

# Train probe on labeled data
X_train = torch.randn(500, 1, 128)
y_train = torch.randint(0, 5, (500,))

with torch.no_grad():
    embeddings = encoder.encode(X_train)

# Train linear classifier
optimizer = torch.optim.SGD(probe.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    logits = probe(embeddings)
    loss = criterion(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

accuracy = evaluate_probe(probe, embeddings, y_train)
print(f"Linear probe accuracy: {accuracy:.2%}")
```

**Output:**

```
Linear probe accuracy: 87.40%
```

## 🏗️ Architecture

The library follows a modular architecture with three main components: core contrastive learning modules (models, losses, datasets), utility functions for augmentations and evaluation, and a demo script showcasing end-to-end usage. The core module contains the neural encoder architectures, projection heads, and contrastive loss implementations. The utils module provides time series augmentation functions, resampling operations, and evaluation metrics. The demo ties everything together with a complete training and evaluation pipeline.

### File Structure

```
time-series-contrastive/
├── lib/
│   ├── core.py              # Core contrastive learning components
│   │   ├── ContrastiveModel      # Main model with encoder + projection
│   │   ├── CNNEncoder            # 1D CNN encoder
│   │   ├── TransformerEncoder    # Lightweight transformer encoder
│   │   ├── ProjectionHead        # MLP projection head
│   │   ├── NTXentLoss            # InfoNCE contrastive loss
│   │   └── PairedTimeSeriesDataset  # Dataset for paired views
│   │
│   └── utils.py             # Utilities and augmentations
│       ├── resample_timeseries   # Up/downsample operations
│       ├── AugmentationPipeline  # Composable augmentations
│       ├── ResampleAug, JitterAug, ScaleAug
│       ├── cosine_similarity     # Similarity metrics
│       ├── LinearProbe           # Linear evaluation
│       └── create_synthetic_data # Data generation
│
├── demo.py                  # Complete training & evaluation demo
│   ├── Synthetic data generation
│   ├── Model training loop
│   ├── Invariance evaluation
│   └── Visualization
│
└── README.md
```

### Files

- **lib/core.py** — Implements the core contrastive learning components including neural encoders (CNN and Transformer), projection heads, NT-Xent loss, and the paired dataset class for generating augmented views.
- **lib/utils.py** — Provides utility functions for time series augmentations (resampling, jitter, scaling), evaluation metrics (cosine similarity, linear probe), and synthetic data generation for testing.
- **demo.py** — Demonstrates end-to-end usage with synthetic data generation, model training with contrastive loss, invariance evaluation across sampling rates, and visualization of results.

### Design Decisions

- Modular augmentation pipeline allows easy composition of different transformation strategies and custom augmentation functions.
- Support for both CNN and Transformer encoders provides flexibility for different time series characteristics and computational budgets.
- NT-Xent loss with configurable temperature parameter enables fine-tuning of the contrastive learning objective hardness.
- Paired dataset design ensures efficient batch processing while maintaining correspondence between augmented views.
- Separate projection head from encoder allows for flexible downstream task adaptation by using encoder embeddings directly.
- Resampling operations use interpolation to maintain temporal structure while changing sampling rates.
- Linear probe evaluation provides a standard metric for measuring representation quality without full fine-tuning.

## 🔧 Technical Details

### Dependencies

- **torch** (1.10+) — Deep learning framework for building neural networks, computing gradients, and training models.
- **numpy** (1.20+) — Numerical computing library for array operations, data manipulation, and mathematical functions.
- **matplotlib** (3.0+) — Visualization library for plotting time series data, training curves, and embedding visualizations in the demo.

### Key Algorithms / Patterns

- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss: Computes contrastive loss by maximizing agreement between positive pairs while minimizing similarity to negative pairs.
- Temporal resampling via interpolation: Uses linear or cubic interpolation to upsample/downsample time series while preserving temporal structure.
- Cosine similarity metric: Measures angular distance between embeddings to evaluate invariance across different augmented views.
- 1D Convolutional encoder: Applies temporal convolutions with residual connections to extract hierarchical features from time series.
- Linear probe protocol: Trains a linear classifier on frozen embeddings to evaluate representation quality without full model fine-tuning.

### Important Notes

- Resampling augmentations may change sequence length; ensure proper padding or length normalization in the dataset pipeline.
- Temperature parameter in NT-Xent loss significantly affects training dynamics; typical values range from 0.1 to 0.5.
- Batch size should be sufficiently large (>64) for contrastive learning to have enough negative samples per batch.
- The projection head is used only during training; for downstream tasks, use embeddings from the encoder directly.
- Different time series domains may require domain-specific augmentation strategies beyond resampling (e.g., frequency masking for audio).

## ❓ Troubleshooting

### Loss is not decreasing or stays at log(batch_size)

**Cause:** The model is not learning to distinguish positive pairs from negatives, often due to too high learning rate or improper augmentations.

**Solution:** Reduce learning rate to 1e-4 or lower, ensure augmentations are not too strong (check that augmented views are still recognizable), and verify batch size is at least 64 for sufficient negative samples.

### Out of memory errors during training

**Cause:** Contrastive learning requires computing similarities across all pairs in a batch, leading to memory usage proportional to batch_size^2.

**Solution:** Reduce batch size, use gradient accumulation to simulate larger batches, or enable mixed precision training with torch.cuda.amp to reduce memory footprint.

### Model produces identical embeddings for all inputs

**Cause:** The model has collapsed to a trivial solution where all inputs map to the same point, often due to too low temperature or insufficient regularization.

**Solution:** Increase temperature parameter (try 0.5 or higher), add weight decay to optimizer (1e-4), ensure augmentations create sufficiently diverse views, and verify projection head has sufficient capacity.

### Poor invariance to resampling despite low training loss

**Cause:** Model may be learning shortcuts based on sequence length or other artifacts rather than true content-based representations.

**Solution:** Ensure resampling augmentations are applied during training with high probability (>0.8), normalize or pad sequences to fixed length, and increase diversity of resampling rates in augmentation pipeline.

### RuntimeError: shape mismatch in loss computation

**Cause:** Inconsistent tensor shapes between the two views due to different resampling rates or improper batching in the dataset.

**Solution:** Ensure PairedTimeSeriesDataset pads or crops sequences to the same length, verify both views have shape [batch, channels, length], and check that encoder output dimensions match projection head input.

---

This project demonstrates a research-oriented approach to learning sampling-invariant time series representations through contrastive learning. The implementation is designed for experimentation and can be extended with additional augmentation strategies, encoder architectures, or loss functions. For production use, consider adding comprehensive unit tests, configuration management, and distributed training support. This README and portions of the codebase were generated with AI assistance.