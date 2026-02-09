"""Utility functions for time series resampling and data processing.

This module provides helper functions for data preparation, batch processing,
and evaluation of resampling-invariant models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Union, Callable
from scipy import signal
import warnings


class ResamplingDataset(Dataset):
    """PyTorch Dataset that generates resampled views on-the-fly.
    
    This dataset wraps time series data and generates multiple resampled
    views for contrastive learning during training.
    
    Attributes:
        data: Time series data array.
        augmenter: ResamplingAugmenter instance for creating views.
        n_views: Number of views to generate per sample.
        target_length: Target length for all views.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        augmenter,
        n_views: int = 2,
        target_length: Optional[int] = None,
        labels: Optional[np.ndarray] = None
    ):
        """Initialize the resampling dataset.
        
        Args:
            data: Time series data of shape (n_samples, length, features) or
                 (n_samples, length).
            augmenter: ResamplingAugmenter instance.
            n_views: Number of views to generate per sample.
            target_length: Target length for resampled views.
            labels: Optional labels for the data.
        """
        self.data = data
        self.augmenter = augmenter
        self.n_views = n_views
        self.target_length = target_length
        self.labels = labels
        
        # Ensure data is 3D: (n_samples, length, features)
        if self.data.ndim == 2:
            self.data = self.data[:, :, np.newaxis]
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, ...], Tuple[Tuple[torch.Tensor, ...], int]]:
        """Get a sample with multiple resampled views.
        
        Args:
            idx: Index of the sample.
        
        Returns:
            Tuple of view tensors, or (views, label) if labels provided.
        """
        x = self.data[idx]
        
        # Generate views
        views = self.augmenter.create_views(
            x,
            n_views=self.n_views,
            target_length=self.target_length
        )
        
        # Convert to tensors and transpose to (features, length)
        view_tensors = []
        for view in views:
            if view.ndim == 1:
                view = view[:, np.newaxis]
            # Transpose to (features, length) for Conv1d
            view_tensor = torch.FloatTensor(view.T)
            view_tensors.append(view_tensor)
        
        if self.labels is not None:
            return tuple(view_tensors), int(self.labels[idx])
        else:
            return tuple(view_tensors)


def create_dataloader(
    data: np.ndarray,
    augmenter,
    batch_size: int = 32,
    n_views: int = 2,
    target_length: Optional[int] = None,
    labels: Optional[np.ndarray] = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for resampling contrastive learning.
    
    Args:
        data: Time series data array.
        augmenter: ResamplingAugmenter instance.
        batch_size: Batch size.
        n_views: Number of views per sample.
        target_length: Target length for views.
        labels: Optional labels.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes.
    
    Returns:
        PyTorch DataLoader.
    """
    dataset = ResamplingDataset(
        data=data,
        augmenter=augmenter,
        n_views=n_views,
        target_length=target_length,
        labels=labels
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


def normalize_timeseries(
    x: np.ndarray,
    method: str = 'zscore',
    axis: int = 0,
    epsilon: float = 1e-8
) -> np.ndarray:
    """Normalize time series data.
    
    Args:
        x: Input time series array.
        method: Normalization method ('zscore', 'minmax', 'robust').
        axis: Axis along which to normalize.
        epsilon: Small constant for numerical stability.
    
    Returns:
        Normalized time series.
    """
    if method == 'zscore':
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        return (x - mean) / (std + epsilon)
    
    elif method == 'minmax':
        min_val = np.min(x, axis=axis, keepdims=True)
        max_val = np.max(x, axis=axis, keepdims=True)
        return (x - min_val) / (max_val - min_val + epsilon)
    
    elif method == 'robust':
        median = np.median(x, axis=axis, keepdims=True)
        q75 = np.percentile(x, 75, axis=axis, keepdims=True)
        q25 = np.percentile(x, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        return (x - median) / (iqr + epsilon)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_embedding_similarity(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    metric: str = 'cosine'
) -> np.ndarray:
    """Compute pairwise similarity between embeddings.
    
    Args:
        embeddings1: First set of embeddings, shape (n, dim).
        embeddings2: Second set of embeddings, shape (m, dim).
        metric: Similarity metric ('cosine', 'euclidean').
    
    Returns:
        Similarity matrix of shape (n, m).
    """
    if metric == 'cosine':
        # Normalize embeddings
        norm1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        norm2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
        return np.dot(norm1, norm2.T)
    
    elif metric == 'euclidean':
        # Compute negative Euclidean distance (higher is more similar)
        distances = np.sqrt(
            np.sum(embeddings1**2, axis=1, keepdims=True) +
            np.sum(embeddings2**2, axis=1, keepdims=True).T -
            2 * np.dot(embeddings1, embeddings2.T)
        )
        return -distances
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate_resampling_invariance(
    model: torch.nn.Module,
    data: np.ndarray,
    augmenter,
    resample_rates: List[float],
    target_length: int,
    device: str = 'cpu'
) -> dict:
    """Evaluate model's invariance to resampling.
    
    Computes the similarity between embeddings of original and resampled
    versions of the same time series.
    
    Args:
        model: Trained encoder model.
        data: Time series data to evaluate.
        augmenter: ResamplingAugmenter instance.
        resample_rates: List of resampling rates to test.
        target_length: Target length for all resampled versions.
        device: Device to run evaluation on.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    model.to(device)
    
    similarities = {rate: [] for rate in resample_rates}
    
    with torch.no_grad():
        for i in range(len(data)):
            x = data[i]
            
            # Get original embedding (at rate 1.0)
            x_original = augmenter.resample_timeseries(x, 1.0)
            x_original = augmenter.resample_timeseries(x_original, target_length / len(x_original))
            
            if x_original.ndim == 1:
                x_original = x_original[:, np.newaxis]
            
            x_tensor = torch.FloatTensor(x_original.T).unsqueeze(0).to(device)
            emb_original = model.encode(x_tensor).cpu().numpy()
            
            # Compare with resampled versions
            for rate in resample_rates:
                x_resampled = augmenter.resample_timeseries(x, rate)
                x_resampled = augmenter.resample_timeseries(
                    x_resampled,
                    target_length / len(x_resampled)
                )
                
                if x_resampled.ndim == 1:
                    x_resampled = x_resampled[:, np.newaxis]
                
                x_tensor = torch.FloatTensor(x_resampled.T).unsqueeze(0).to(device)
                emb_resampled = model.encode(x_tensor).cpu().numpy()
                
                # Compute cosine similarity
                sim = compute_embedding_similarity(emb_original, emb_resampled, metric='cosine')[0, 0]
                similarities[rate].append(sim)
    
    # Compute statistics
    results = {
        'mean_similarity': {rate: np.mean(sims) for rate, sims in similarities.items()},
        'std_similarity': {rate: np.std(sims) for rate, sims in similarities.items()},
        'min_similarity': {rate: np.min(sims) for rate, sims in similarities.items()},
        'max_similarity': {rate: np.max(sims) for rate, sims in similarities.items()},
    }
    
    return results


def add_noise(
    x: np.ndarray,
    noise_level: float = 0.01,
    noise_type: str = 'gaussian'
) -> np.ndarray:
    """Add noise to time series data.
    
    Args:
        x: Input time series.
        noise_level: Standard deviation of noise relative to signal std.
        noise_type: Type of noise ('gaussian', 'uniform').
    
    Returns:
        Noisy time series.
    """
    signal_std = np.std(x)
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level * signal_std, x.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(
            -noise_level * signal_std,
            noise_level * signal_std,
            x.shape
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return x + noise


def split_train_val_test(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, validation, and test sets.
    
    Args:
        data: Input data array.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        test_ratio: Proportion for test set.
        shuffle: Whether to shuffle before splitting.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    return data[train_indices], data[val_indices], data[test_indices]


def _pad_or_truncate(
    x: np.ndarray,
    target_length: int,
    pad_value: float = 0.0
) -> np.ndarray:
    """Helper function to pad or truncate time series to target length.
    
    Args:
        x: Input time series.
        target_length: Desired length.
        pad_value: Value to use for padding.
    
    Returns:
        Padded or truncated time series.
    """
    current_length = len(x)
    
    if current_length == target_length:
        return x
    elif current_length > target_length:
        # Truncate
        return x[:target_length]
    else:
        # Pad
        if x.ndim == 1:
            pad_width = (0, target_length - current_length)
        else:
            pad_width = ((0, target_length - current_length), (0, 0))
        return np.pad(x, pad_width, mode='constant', constant_values=pad_value)
