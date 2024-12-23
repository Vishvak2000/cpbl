import numpy as np
import matplotlib.pyplot as plt
import torch

def get_attention_weights(model):
    """Return the attention weights from the last forward pass of the model."""
    if model.use_attention_pooling and hasattr(model, 'last_attention_weights'):
        return model.last_attention_weights
    return None

def visualize_attention(model, sequence_idx=0, batch_idx=0):
    """
    Visualize the attention weights for a specific sequence in the batch.
    Parameters:
    - model: the BPNetLightning model instance
    - sequence_idx: the index of the sequence to visualize within the batch
    - batch_idx: the index of the batch
    """
    weights = get_attention_weights(model)
    if weights is None:
        print("Attention pooling is not used or weights are not available.")
        return None
    
    # Detach and extract specific sequence weights
    weights = weights.detach()
    sequence_weights = weights[batch_idx, :, 0].cpu().numpy()  # Shape: [batch_size, sequence_length, 1]

    # Plot attention weights
    plt.figure(figsize=(10, 4))
    plt.plot(sequence_weights)
    plt.title(f"Attention Weights for Sequence {sequence_idx} in Batch {batch_idx}")
    plt.xlabel("Position in Sequence")
    plt.ylabel("Attention Weight")
    plt.show()
    
    return sequence_weights
