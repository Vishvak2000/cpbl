import torch
import torch.distributions as dist

def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values (torch.Tensor)
      logits: predicted logit values (torch.Tensor)
    """
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)
    
    # Sum of counts across the last dimension
    counts_per_example = torch.sum(true_counts, dim=-1)
    
    # Create a Multinomial distribution object
    m_dist = dist.Multinomial(total_count=counts_per_example, probs=probabilities)
    
    # Calculate the negative log likelihood
    log_probs = m_dist.log_prob(true_counts)
    nll = -torch.sum(log_probs) / true_counts.shape[0]
    
    return nll
