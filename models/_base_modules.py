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

class GlobalAvgPool1D(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x has shape (batch_size, channels, time_steps) in PyTorch
        return torch.mean(x, dim=2)

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

## Can try an RNN, etc
    
class MLP_regressor(nn.Module):
    def __init__(self, n_input: int,
                 n_hidden: int,
                 n_layers: int,
                 n_output: int,
                 activation_fn: nn.Module
                 ):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers


        self.activation_fn = activation_fn

        layers = [n_input] + [n_hidden for _ in range(n_layers)]

        self.network = nn.ModuleList()
        for n_in, n_out in zip(layers[:-1], layers[1:]):
            self.network.append(
                nn.Linear(n_in, n_out, bias=True)
            )
            self.network.append(
                self.activation_fn()
            )
    
        self.regressor = nn.Linear(n_hidden, n_output)
        self.regressor = self.regressor
        
        self.network = nn.Sequential(*self.network)
        self.network = self.network