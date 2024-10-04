#here is where the initial modules are eg: CNN, RNN, etc
import torch
import torch.nn as nn


class CNNModule(nn.Module): #set up a base convolutional layer, for the first or the post dilation layers
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

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DilatedConvModule(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size, 
                 dilation, 
                 padding=0, 
                 activation_fn=nn.ReLU):
        super().__init__()
        padding = (kernel_size - 1) * dilation //2 # this is new
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.activation = activation_fn()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Cropping1D(nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size  # This will be the total amount to crop (both sides combined)

    def forward(self, x):
        # Calculate left and right crop amounts
        crop_left = self.crop_size // 2
        crop_right = self.crop_size - crop_left
        
        # Return the cropped tensor
        return x[:, :, crop_left:-crop_right] if crop_right > 0 else x[:, :, crop_left:]


class Cropping1D_old(nn.Module):
    def __init__(self, cropsize):
        super().__init__()
        self.cropsize = cropsize  # This would define how much to crop on each side

    def forward(self, x):
        # Assume prof_out_precrop has shape (batch_size, time_steps, channels)
        steps = x.shape[1]

        # Cropping based on the cropsize. Crops the first and last 'cropsize' elements from the time_steps dimension
        cropped_output = x[:, self.cropsize:steps - self.cropsize, :]

        return cropped_output

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
        return torch.mean(x, dim=1)

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