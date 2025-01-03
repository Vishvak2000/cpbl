#here is where the initial modules are eg: CNN, RNN, etc
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModule(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0,
                 activation_fn=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                            out_channels=out_channels, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding)
        self.activation = activation_fn()
        #self.bn = nn.BatchNorm1d(out_channels)  # Added batch normalization

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)  # Apply batch normalization
        x = self.activation(x)
        return x

class DilatedConvModule(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 dilation, 
                 padding=0, 
                 activation_fn=nn.ReLU):
        super().__init__()
        # Remove automatic padding calculation to match 'valid' padding behavior
        self.conv = nn.Conv1d(in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=kernel_size, 
                            dilation=dilation, 
                            padding=padding)
        self.activation = activation_fn()
        #self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)  # Apply batch normalization
        x = self.activation(x)
        return x


class Cropping1D(nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size
    
    def forward(self, x):
        # Calculate the amount to crop from each side
        crop_per_side = self.crop_size // 2
        return x[:, :, crop_per_side:-crop_per_side]
    
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # Equivalent to Flatten() in Keras

    def forward(self, x):
        # Apply the flatten operation to prof
        x = self.flatten(x)
        return x


class GlobalAvgPool1D(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x has shape (batch_size, time_steps, channels)
        # Perform global average pooling across the time_steps (dim=1)
        return torch.mean(x, dim=-1)


class AttentionPooling(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        if hidden_features is None:
            hidden_features = in_features // 2
            
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),  # Tanh gives better stability than ReLU for attention
            nn.Linear(hidden_features, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length]
        # Transpose to [batch_size, sequence_length, channels]
        x = x.transpose(1, 2)
        
        # Calculate attention weights
        weights = self.attention(x)  # [batch_size, sequence_length, 1]
        weights = F.softmax(weights, dim=1)  # Normalize across sequence length
        
        # Apply attention weights
        weighted = x * weights  # Broadcasting will handle the dimensions
        
        # Sum across sequence length
        pooled = weighted.sum(dim=1)  # [batch_size, channels]
        
        return pooled, weights