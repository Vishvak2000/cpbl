import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class MultinomialNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, true_counts):
        # Compute log probabilities for each category
        true_counts = torch.clamp(true_counts, min=1e-8)  # Avoid log(0)
        log_probs = F.log_softmax(logits, dim=-1)
        log_likelihood = torch.sum(true_counts * log_probs, dim=-1)
        total_counts = true_counts.sum(dim=-1)
        log_gamma_term = torch.lgamma(total_counts + 1)
        log_gamma_counts = torch.lgamma(true_counts + 1).sum(dim=-1)
        log_likelihood = log_likelihood + log_gamma_term - log_gamma_counts
        return -torch.mean(log_likelihood)


class CombinedLoss(nn.Module):
    def __init__(self, avg_total_counts=None, alpha=1.0, flavor=None):
        """
        Combined loss function
        
        Args:
        - avg_total_counts: Average total counts across training set for lambda calculation
        - alpha: Scaling factor for total count loss (default 1.0)
        """
        super().__init__()

        if flavor == "mse":
            self.multinomial_loss = nn.MSELoss(reduction="mean")
            self.mse_loss = nn.MSELoss(reduction="mean")
        else: 
            self.multinomial_loss = MultinomialNLLLoss()
            self.mse_loss = nn.MSELoss(reduction="mean")
        
        if avg_total_counts is not None:
            self.lambda_weight = alpha * (avg_total_counts / 2)
        else:
            self.lambda_weight = 1.0
        
        self.alpha = alpha
    
    def forward(self, predictions, targets):
        """
        Compute combined loss
        
        Args:
        - predictions: [log_total_counts, profile_predictions ]
        - targets: [total_counts, profile_targets]
        """
        log_total_count_pred, profile_pred = predictions
        log_total_count_target, profile_target = targets
        #print(f"predicted counts shape = {log_total_count_pred.shape}")
        #print(f"true counts shape = {log_total_count_target.shape}")

        #print(f"predicted profile shape = {profile_pred.shape}")
        #print(f"true profile shape = {profile_target.shape}")

        
        # Multinomial loss for profile prediction
        multinomial_loss = self.multinomial_loss(profile_pred, profile_target)
        
        count_loss = self.mse_loss(log_total_count_pred,log_total_count_target)
        
        # Combined loss with alpha scaling
        total_loss = multinomial_loss + self.lambda_weight * count_loss
        
        return total_loss

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
        seq_len = predictions.shape[1]
        
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

def spearman_corrcoef(y_pred, y_true):
    """
    Calculates the Spearman rank correlation coefficient.
    Args:
        y_pred (Tensor): Predicted values, shape [batch_size, num_features].
        y_true (Tensor): True values, shape [batch_size, num_features].
    Returns:
        Tensor: Spearman correlation coefficient for each feature.
    """
    # Rank the values along each feature dimension
    rank_pred = torch.argsort(torch.argsort(y_pred, dim=0), dim=0).float() + 1
    rank_true = torch.argsort(torch.argsort(y_true, dim=0), dim=0).float() + 1
    
    # Compute the differences in ranks
    d = rank_pred - rank_true
    
    # Compute Spearman correlation using the formula
    n = y_pred.size(0)  # Number of samples (batch size)
    rho = 1 - (6 * torch.sum(d ** 2, dim=0)) / (n * (n ** 2 - 1))
    
    return rho