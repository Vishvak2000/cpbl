import torch
import torch.nn as nn
import numpy as np

class WeightedMSELoss(nn.Module):
    def __init__(self, sequence_length, center_size=500, center_weight=10.0, sigma_scale=0.25):
        super().__init__()
        self.sequence_length = sequence_length
        self.center_size = center_size
        self.center_weight = center_weight
        self.sigma_scale = sigma_scale
        self.weights = self._create_weights()

    def _create_weights(self):
        x = np.arange(self.sequence_length)
        mu = self.sequence_length / 2
        sigma = self.center_size * self.sigma_scale
        
        # Create Gaussian distribution
        gaussian = np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        # Normalize Gaussian to range [0, 1]
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
        
        # Scale Gaussian to range [1, center_weight]
        weights = 1 + (self.center_weight - 1) * gaussian
        
        return torch.tensor(weights, dtype=torch.float32)

    def forward(self, predictions, targets):
        squared_diff = (predictions - targets) ** 2
        weighted_squared_diff = squared_diff * self.weights
        return torch.mean(weighted_squared_diff)
    
class WeightedMSELossNorm(nn.Module):
    def __init__(self, sequence_length, center_size=500, center_weight=10.0, sigma_scale=0.25):
        super().__init__()
        self.sequence_length = sequence_length
        self.center_size = center_size
        self.center_weight = center_weight
        self.sigma_scale = sigma_scale
        self.weights = self._create_weights()

    def _create_weights(self):
        x = np.arange(self.sequence_length)
        mu = self.sequence_length / 2
        sigma = self.center_size * self.sigma_scale
        
        # Create Gaussian distribution
        gaussian = np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        # Normalize Gaussian to range [0, 1]
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
        
        # Scale Gaussian to range [1, center_weight]
        weights = 1 + (self.center_weight - 1) * gaussian
        
        return torch.tensor(weights, dtype=torch.float32)

    def forward(self, predictions, targets):
        squared_diff = (predictions - targets) ** 2
        weighted_squared_diff = squared_diff * self.weights
        return torch.mean(weighted_squared_diff)/ self.weights.sum()
    

class FocusedMSELoss(nn.Module):
    def __init__(self, n):
        """
        Args:
            n (int): Number of middle sequences to consider for MSE.
        """
        super().__init__()
        self.n = n  # Number of middle sequences to calculate MSE on
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): Predicted values of shape (batch_size, sequence_length)
            targets (torch.Tensor): Ground truth values of shape (batch_size, sequence_length)
        
        Returns:
            torch.Tensor: MSE calculated on the middle `n` sequences.
        """
        # Get the total length of the sequences
        seq_len = predictions.size(1)
        
        # Ensure that n is not larger than the sequence length
        if self.n > seq_len:
            raise ValueError(f"n ({self.n}) cannot be larger than the sequence length ({seq_len})")
        
        # Calculate start and end indices for the middle n sequences
        start_idx = (seq_len - self.n) // 2
        end_idx = start_idx + self.n
        
        # Extract the middle n sequences from predictions and targets
        middle_predictions = predictions[:, start_idx:end_idx]
        middle_targets = targets[:, start_idx:end_idx]
        
        # Compute MSE on the middle n sequences
        mse = torch.mean((middle_predictions - middle_targets) ** 2)
        
        return mse
