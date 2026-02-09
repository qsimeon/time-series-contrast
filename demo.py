"""
Demo Script: Resampling-Invariant Time Series Contrastive Learning

This script demonstrates how to train a time series model to be invariant to
upsampling and downsampling using contrastive learning. Different views of the
data are created through different resamplings, and the model learns to produce
similar embeddings for these different views.

Key Concepts:
- Resampling augmentation creates different temporal resolutions of the same signal
- Contrastive loss encourages the model to produce similar embeddings for different
  resamplings of the same time series
- This makes the model robust to sampling rate variations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import from the provided modules
from core import (
    ResamplingAugmenter,
    ContrastiveLoss,
    TimeSeriesEncoder,
    ResamplingContrastiveModel
)
from utils import (
    ResamplingDataset,
    create_dataloader,
    normalize_timeseries,
    compute_embedding_similarity,
    evaluate_resampling_invariance,
    add_noise,
    split_train_val_test
)


def generate_synthetic_timeseries(
    n_samples: int = 1000,
    length: int = 200,
    n_features: int = 1,
    signal_types: List[str] = ['sine', 'cosine', 'sawtooth', 'square']
) -> np.ndarray:
    """
    Generate synthetic time series data for demonstration.
    
    Args:
        n_samples: Number of time series samples to generate
        length: Length of each time series
        n_features: Number of features per time point
        signal_types: Types of signals to generate
        
    Returns:
        Array of shape (n_samples, length, n_features)
    """
    data = []
    
    for i in range(n_samples):
        # Randomly select signal type
        signal_type = np.random.choice(signal_types)
        
        # Random frequency and phase
        frequency = np.random.uniform(0.5, 3.0)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.5, 2.0)
        
        t = np.linspace(0, 10, length)
        
        if signal_type == 'sine':
            signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        elif signal_type == 'cosine':
            signal = amplitude * np.cos(2 * np.pi * frequency * t + phase)
        elif signal_type == 'sawtooth':
            signal = amplitude * (2 * (t * frequency - np.floor(t * frequency + 0.5)))
        elif signal_type == 'square':
            signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phase))
        
        # Add some noise
        signal = signal + np.random.normal(0, 0.1, length)
        
        # Reshape for multi-feature support
        if n_features == 1:
            signal = signal.reshape(-1, 1)
        else:
            # Generate multiple correlated features
            signal_multi = np.zeros((length, n_features))
            signal_multi[:, 0] = signal.flatten()
            for j in range(1, n_features):
                signal_multi[:, j] = signal.flatten() + np.random.normal(0, 0.2, length)
            signal = signal_multi
        
        data.append(signal)
    
    return np.array(data)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The contrastive model
        dataloader: DataLoader providing batches
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # batch is a list of views: [view1, view2, ...]
        # Each view has shape (batch_size, length, n_features)
        
        # Move all views to device
        views = [view.to(device).float() for view in batch]
        
        optimizer.zero_grad()
        
        # Get embeddings for all views
        embeddings = [model.encode(view) for view in views]
        
        # Compute contrastive loss between pairs of views
        # Use first two views as anchor and positive
        loss = criterion(embeddings[0], embeddings[1])
        
        # If more than 2 views, add additional contrastive pairs
        if len(embeddings) > 2:
            for i in range(2, len(embeddings)):
                loss += criterion(embeddings[0], embeddings[i])
            loss = loss / (len(embeddings) - 1)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Validate the model for one epoch.
    
    Args:
        model: The contrastive model
        dataloader: DataLoader providing batches
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            views = [view.to(device).float() for view in batch]
            
            embeddings = [model.encode(view) for view in views]
            
            loss = criterion(embeddings[0], embeddings[1])
            
            if len(embeddings) > 2:
                for i in range(2, len(embeddings)):
                    loss += criterion(embeddings[0], embeddings[i])
                loss = loss / (len(embeddings) - 1)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def visualize_embeddings(
    model: nn.Module,
    data: np.ndarray,
    augmenter: ResamplingAugmenter,
    resample_rates: List[float],
    target_length: int,
    device: str,
    n_samples: int = 5
) -> None:
    """
    Visualize how embeddings change with different resampling rates.
    
    Args:
        model: Trained model
        data: Time series data
        augmenter: Resampling augmenter
        resample_rates: List of resampling rates to test
        target_length: Target length for resampled series
        device: Device to run on
        n_samples: Number of samples to visualize
    """
    model.eval()
    
    # Select random samples
    indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            sample = data[sample_idx]
            
            # Generate resampled versions
            resampled_versions = []
            embeddings = []
            
            for rate in resample_rates:
                resampled = augmenter.resample_timeseries(sample, rate, target_length)
                resampled_versions.append(resampled)
                
                # Get embedding
                tensor = torch.from_numpy(resampled).unsqueeze(0).float().to(device)
                embedding = model.encode(tensor).cpu().numpy()
                embeddings.append(embedding.flatten())
            
            # Plot resampled time series
            ax1 = axes[idx, 0]
            for i, (resampled, rate) in enumerate(zip(resampled_versions, resample_rates)):
                ax1.plot(resampled[:, 0], alpha=0.7, label=f'Rate: {rate:.2f}x')
            ax1.set_title(f'Sample {sample_idx}: Resampled Time Series')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot embedding similarities
            ax2 = axes[idx, 1]
            embeddings_array = np.array(embeddings)
            
            # Compute similarity matrix
            similarity_matrix = compute_embedding_similarity(
                embeddings_array, embeddings_array, metric='cosine'
            )
            
            im = ax2.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
            ax2.set_title(f'Sample {sample_idx}: Embedding Similarity Matrix')
            ax2.set_xlabel('Resampling Rate Index')
            ax2.set_ylabel('Resampling Rate Index')
            
            # Add colorbar
            plt.colorbar(im, ax=ax2)
            
            # Add rate labels
            rate_labels = [f'{rate:.2f}x' for rate in resample_rates]
            ax2.set_xticks(range(len(resample_rates)))
            ax2.set_yticks(range(len(resample_rates)))
            ax2.set_xticklabels(rate_labels, rotation=45)
            ax2.set_yticklabels(rate_labels)
    
    plt.tight_layout()
    plt.savefig('resampling_invariance_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to 'resampling_invariance_visualization.png'")
    plt.close()


def main():
    """
    Main demonstration function.
    """
    print("=" * 80)
    print("Resampling-Invariant Time Series Contrastive Learning Demo")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Using device: {device}")
    
    # Hyperparameters
    n_samples = 1000
    ts_length = 200
    n_features = 1
    target_length = 128
    embedding_dim = 64
    hidden_dim = 128
    n_views = 3
    batch_size = 32
    n_epochs = 20
    learning_rate = 0.001
    
    print(f"\n{'Configuration:':<30}")
    print(f"  {'Number of samples:':<28} {n_samples}")
    print(f"  {'Time series length:':<28} {ts_length}")
    print(f"  {'Number of features:':<28} {n_features}")
    print(f"  {'Target length:':<28} {target_length}")
    print(f"  {'Embedding dimension:':<28} {embedding_dim}")
    print(f"  {'Number of views:':<28} {n_views}")
    print(f"  {'Batch size:':<28} {batch_size}")
    print(f"  {'Number of epochs:':<28} {n_epochs}")
    print(f"  {'Learning rate:':<28} {learning_rate}")
    
    # Step 1: Generate synthetic time series data
    print(f"\n{'Step 1: Generating synthetic time series data':<60}", end=" ... ")
    data = generate_synthetic_timeseries(
        n_samples=n_samples,
        length=ts_length,
        n_features=n_features
    )
    print("✓")
    print(f"  Data shape: {data.shape}")
    
    # Step 2: Normalize data
    print(f"\n{'Step 2: Normalizing time series data':<60}", end=" ... ")
    data_normalized = normalize_timeseries(data, method='zscore', axis=1)
    print("✓")
    
    # Step 3: Split data
    print(f"\n{'Step 3: Splitting data into train/val/test sets':<60}", end=" ... ")
    train_data, val_data, test_data = split_train_val_test(
        data_normalized,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True,
        random_seed=42
    )
    print("✓")
    print(f"  Train: {train_data.shape[0]} samples")
    print(f"  Val:   {val_data.shape[0]} samples")
    print(f"  Test:  {test_data.shape[0]} samples")
    
    # Step 4: Create resampling augmenter
    print(f"\n{'Step 4: Creating resampling augmenter':<60}", end=" ... ")
    augmenter = ResamplingAugmenter(
        resample_range=(0.5, 2.0),  # Resample between 0.5x and 2.0x
        method='linear'
    )
    print("✓")
    print(f"  Resampling range: 0.5x to 2.0x")
    print(f"  Interpolation method: linear")
    
    # Step 5: Create data loaders
    print(f"\n{'Step 5: Creating data loaders':<60}", end=" ... ")
    train_loader = create_dataloader(
        train_data,
        augmenter=augmenter,
        batch_size=batch_size,
        n_views=n_views,
        target_length=target_length,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader(
        val_data,
        augmenter=augmenter,
        batch_size=batch_size,
        n_views=n_views,
        target_length=target_length,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = create_dataloader(
        test_data,
        augmenter=augmenter,
        batch_size=batch_size,
        n_views=n_views,
        target_length=target_length,
        shuffle=False,
        num_workers=0
    )
    print("✓")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # Step 6: Initialize model
    print(f"\n{'Step 6: Initializing model':<60}", end=" ... ")
    model = ResamplingContrastiveModel(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_layers=2
    ).to(device)
    print("✓")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    
    # Step 7: Initialize loss and optimizer
    print(f"\n{'Step 7: Initializing loss and optimizer':<60}", end=" ... ")
    criterion = ContrastiveLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("✓")
    print(f"  Loss: Contrastive Loss (temperature=0.5)")
    print(f"  Optimizer: Adam (lr={learning_rate})")
    
    # Step 8: Training loop
    print(f"\n{'Step 8: Training model':<60}")
    print("-" * 80)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    print("-" * 80)
    print(f"✓ Training complete! Best validation loss: {best_val_loss:.4f}")
    
    # Step 9: Load best model and evaluate
    print(f"\n{'Step 9: Evaluating resampling invariance':<60}", end=" ... ")
    model.load_state_dict(torch.load('best_model.pth'))
    
    resample_rates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    invariance_results = evaluate_resampling_invariance(
        model=model,
        data=test_data[:100],  # Use subset for faster evaluation
        augmenter=augmenter,
        resample_rates=resample_rates,
        target_length=target_length,
        device=device
    )
    print("✓")
    
    print(f"\n{'Resampling Invariance Results:':<30}")
    print(f"  {'Mean similarity:':<28} {invariance_results['mean_similarity']:.4f}")
    print(f"  {'Std similarity:':<28} {invariance_results['std_similarity']:.4f}")
    print(f"  {'Min similarity:':<28} {invariance_results['min_similarity']:.4f}")
    print(f"  {'Max similarity:':<28} {invariance_results['max_similarity']:.4f}")
    
    # Step 10: Visualize results
    print(f"\n{'Step 10: Visualizing results':<60}", end=" ... ")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓")
    print(f"  Training curves saved to 'training_curves.png'")
    
    # Visualize embeddings for different resampling rates
    print(f"\n{'Step 11: Visualizing embedding invariance':<60}", end=" ... ")
    visualize_embeddings(
        model=model,
        data=test_data,
        augmenter=augmenter,
        resample_rates=resample_rates,
        target_length=target_length,
        device=device,
        n_samples=3
    )
    print("✓")
    
    # Step 12: Test on specific examples
    print(f"\n{'Step 12: Testing on specific examples':<60}")
    print("-" * 80)
    
    model.eval()
    with torch.no_grad():
        # Take a test sample
        test_sample = test_data[0]
        
        # Create different resampled versions
        print("\nComputing embeddings for different resampling rates:")
        embeddings_list = []
        
        for rate in resample_rates:
            resampled = augmenter.resample_timeseries(test_sample, rate, target_length)
            tensor = torch.from_numpy(resampled).unsqueeze(0).float().to(device)
            embedding = model.encode(tensor).cpu().numpy().flatten()
            embeddings_list.append(embedding)
            print(f"  Rate {rate:.2f}x: embedding shape {embedding.shape}")
        
        # Compute pairwise similarities
        print("\nPairwise cosine similarities:")
        embeddings_array = np.array(embeddings_list)
        
        for i in range(len(resample_rates)):
            for j in range(i+1, len(resample_rates)):
                sim = compute_embedding_similarity(
                    embeddings_array[i:i+1],
                    embeddings_array[j:j+1],
                    metric='cosine'
                )[0, 0]
                print(f"  {resample_rates[i]:.2f}x vs {resample_rates[j]:.2f}x: {sim:.4f}")
    
    print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Successfully trained a resampling-invariant time series model")
    print(f"✓ Model achieves {invariance_results['mean_similarity']:.2%} average similarity")
    print(f"  across different resampling rates")
    print(f"✓ This demonstrates that the model has learned representations that are")
    print(f"  robust to temporal resolution changes (upsampling/downsampling)")
    print(f"\nKey Insights:")
    print(f"  • Contrastive learning with resampling augmentation creates invariance")
    print(f"  • Different temporal resolutions of the same signal produce similar embeddings")
    print(f"  • This is useful for real-world scenarios with varying sampling rates")
    print("=" * 80)
    
    print("\n✓ Demo completed successfully!")
    print(f"  Generated files:")
    print(f"    - best_model.pth")
    print(f"    - training_curves.png")
    print(f"    - resampling_invariance_visualization.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
