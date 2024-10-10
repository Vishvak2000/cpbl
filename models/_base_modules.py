#here is where the initial modules are eg: CNN, RNN, etc
import torch
import torch.nn as nn


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
        self.crop_size = crop_size  # Total amount to crop (both sides combined)

    def forward(self, x):
        # For skip connections, we need to crop from both ends equally
        crop_left = self.crop_size // 2
        crop_right = self.crop_size - crop_left
        
        if crop_right > 0:
            return x[:, :, crop_left:-crop_right]
        return x[:, :, crop_left:]

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        # Assign the flatten operation to an attribute (which acts like a "name")
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
