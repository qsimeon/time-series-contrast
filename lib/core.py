"""Core module for time series contrastive learning with resampling invariance.

This module provides the main components for training models that are invariant
to upsampling and downsampling of time series data through contrastive learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Callable
from scipy import signal
from scipy.interpolate import interp1d


class ResamplingAugmenter:
    """Augmenter that creates different views of time series through resampling.
    
    This class generates multiple views of the same time series by applying
    different resampling rates, creating positive pairs for contrastive learning.
    
    Attributes:
        resample_rates: List of resampling rates to apply (e.g., [0.5, 1.0, 2.0]).
        interpolation_method: Method for interpolation ('linear', 'cubic', 'nearest').
    """
    
    def __init__(
        self,
        resample_rates: List[float] = [0.5, 0.75, 1.0, 1.5, 2.0],
        interpolation_method: str = 'linear'
    ):
        """Initialize the resampling augmenter.
        
        Args:
            resample_rates: List of resampling factors. Values < 1 downsample,
                          values > 1 upsample, 1.0 keeps original.
            interpolation_method: Interpolation method for resampling.
        """
        self.resample_rates = resample_rates
        self.interpolation_method = interpolation_method
    
    def resample_timeseries(
        self,
        x: np.ndarray,
        rate: float
    ) -> np.ndarray:
        """Resample a time series by a given rate.
        
        Args:
            x: Input time series of shape (length, features) or (length,).
            rate: Resampling rate. < 1 downsamples, > 1 upsamples.
        
        Returns:
            Resampled time series.
        """
        if rate == 1.0:
            return x
        
        original_length = len(x)
        new_length = int(original_length * rate)
        
        if new_length < 2:
            new_length = 2
        
        # Handle both 1D and 2D arrays
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            squeeze = True
        else:
            squeeze = False
        
        # Create interpolation function
        old_indices = np.linspace(0, original_length - 1, original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)
        
        resampled = np.zeros((new_length, x.shape[1]))
        for i in range(x.shape[1]):
            f = interp1d(old_indices, x[:, i], kind=self.interpolation_method)
            resampled[:, i] = f(new_indices)
        
        if squeeze:
            resampled = resampled.squeeze()
        
        return resampled
    
    def create_views(
        self,
        x: np.ndarray,
        n_views: int = 2,
        target_length: Optional[int] = None
    ) -> List[np.ndarray]:
        """Create multiple resampled views of a time series.
        
        Args:
            x: Input time series of shape (length, features) or (length,).
            n_views: Number of views to generate.
            target_length: If specified, all views will be resampled to this length.
        
        Returns:
            List of resampled views.
        """
        views = []
        selected_rates = np.random.choice(self.resample_rates, size=n_views, replace=True)
        
        for rate in selected_rates:
            resampled = self.resample_timeseries(x, rate)
            
            # Optionally resample to target length
            if target_length is not None:
                current_length = len(resampled)
                final_rate = target_length / current_length
                resampled = self.resample_timeseries(resampled, final_rate)
            
            views.append(resampled)
        
        return views


class ContrastiveLoss(nn.Module):
    """Contrastive loss for time series representations.
    
    Implements NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
    for contrastive learning with multiple positive pairs.
    
    Attributes:
        temperature: Temperature parameter for scaling similarities.
        reduction: Reduction method ('mean' or 'sum').
    """
    
    def __init__(self, temperature: float = 0.5, reduction: str = 'mean'):
        """Initialize contrastive loss.
        
        Args:
            temperature: Temperature scaling parameter.
            reduction: How to reduce the loss ('mean' or 'sum').
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss between two views.
        
        Args:
            z_i: Embeddings from view i, shape (batch_size, embedding_dim).
            z_j: Embeddings from view j, shape (batch_size, embedding_dim).
        
        Returns:
            Contrastive loss value.
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )
        
        # Create mask for positive pairs
        mask = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)
        mask = mask.repeat(2, 2)
        
        # Mask out self-similarities
        mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask_self, -float('inf'))
        
        # Extract positive pairs
        positives = similarity_matrix[mask].view(2 * batch_size, 1)
        
        # Extract negative pairs
        negatives = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # Concatenate positives and negatives
        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature
        
        # Labels: positive pair is always at index 0
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss


class TimeSeriesEncoder(nn.Module):
    """Base encoder for time series data.
    
    A simple CNN-based encoder that processes time series and produces
    fixed-size embeddings.
    
    Attributes:
        input_channels: Number of input channels/features.
        hidden_dim: Hidden dimension size.
        output_dim: Output embedding dimension.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        sequence_length: int = 100
    ):
        """Initialize time series encoder.
        
        Args:
            input_channels: Number of input features/channels.
            hidden_dim: Size of hidden layers.
            output_dim: Size of output embedding.
            sequence_length: Expected length of input sequences.
        """
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode time series to embedding.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length).
        
        Returns:
            Embedding tensor of shape (batch_size, output_dim).
        """
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Projection
        x = self.projection(x)
        
        return x


class ResamplingContrastiveModel(nn.Module):
    """Complete model for resampling-invariant contrastive learning.
    
    This model combines an encoder with contrastive loss to learn
    representations invariant to time series resampling.
    
    Attributes:
        encoder: Time series encoder network.
        loss_fn: Contrastive loss function.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        temperature: float = 0.5
    ):
        """Initialize the contrastive model.
        
        Args:
            encoder: Encoder network for time series.
            temperature: Temperature for contrastive loss.
        """
        super().__init__()
        self.encoder = encoder
        self.loss_fn = ContrastiveLoss(temperature=temperature)
    
    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with two views.
        
        Args:
            view1: First view, shape (batch_size, channels, length).
            view2: Second view, shape (batch_size, channels, length).
        
        Returns:
            Tuple of (loss, embedding1, embedding2).
        """
        # Encode both views
        z1 = self.encoder(view1)
        z2 = self.encoder(view2)
        
        # Compute contrastive loss
        loss = self.loss_fn(z1, z2)
        
        return loss, z1, z2
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a time series without computing loss.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length).
        
        Returns:
            Embedding tensor.
        """
        return self.encoder(x)
