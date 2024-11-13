import torch
import torch.nn as nn
import numpy as np

class WeightedMSELossNorm(nn.Module):
    def __init__(self, sequence_length, center_size=500, center_weight=10.0, sigma_scale=0.25,scale=False):
        super().__init__()
        self.sequence_length = sequence_length
        self.center_size = center_size
        self.center_weight = center_weight
        self.sigma_scale = sigma_scale
        self.scale = scale
        self.register_buffer('positions', torch.arange(self.sequence_length, dtype=torch.float32))

    def _create_weights(self, summit_indices):
        batch_size = summit_indices.shape[0]
        positions = self.positions.unsqueeze(0).expand(batch_size, -1)
        mu = summit_indices.float().unsqueeze(1)
        
        sigma = self.center_size * self.sigma_scale
        gaussian = torch.exp(-(positions - mu)**2 / (2 * sigma**2))
        
        # Make weights much more gentle - reduce center_weight influence
        weights = 1 + (self.center_weight - 1) * gaussian
        
        # Normalize weights to sum to 1 for each sample
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        return weights

    def forward(self, predictions, targets, debug=False):
        summit_indices = torch.argmax(targets, dim=1)
        weights = self._create_weights(summit_indices)
        weights = weights.to(predictions.device)
        
        squared_diff = (predictions - targets) ** 2
        weighted_errors = squared_diff * weights
       
        # Sum over sequence (since weights sum to 1, this is like a weighted average)
        sample_losses = torch.sum(weighted_errors, dim=1)
        
        # Average over batch
        loss = torch.mean(sample_losses)
        
        if self.scale:
            loss = loss/1e6
        
        return loss

class FocusedMSELoss(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, predictions, targets):
        seq_len = predictions.size(1)
        if self.n > seq_len:
            raise ValueError(f"n ({self.n}) cannot be larger than the sequence length ({seq_len})")
        start_idx = (seq_len - self.n) // 2
        end_idx = start_idx + self.n
        middle_predictions = predictions[:, start_idx:end_idx]
        middle_targets = targets[:, start_idx:end_idx]
        mse = torch.mean((middle_predictions - middle_targets) ** 2)
        return mse
