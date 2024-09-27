#here is where the initial modules are eg: CNN, RNN, etc
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,
                 n_filters: int,
                 in_channels: int = 4,
                 n_layers: int = 3,
                 conv1_kernel_size: int = 21,
                 kernel_size: int = 3,
                 #use_batch_norm: bool = False,
                 #use_layer_norm: bool = False,
                 #dropout_rate: float = 0.0,
                 pooling: str = 'max',
                 pooling_size: int = 2,
                 activation_fn: nn.Module = nn.ReLU,
                 dilation_rates=None
                 ):
        
        assert pooling.lower() in ['max', 'avg']
        super().__init__()

        self.in_channels = in_channels
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.conv1_kernel_size = conv1_kernel_size
        #self.use_batch_norm = use_batch_norm
        #self.use_layer_norm = use_layer_norm
        #self.dropout_rate = dropout_rate
        self.pooling = nn.MaxPool1d if pooling == 'max' else nn.AvgPool1d
        self.pooling_size = pooling_size
        self.activation_fn = activation_fn
        self.dilation_rates = dilation_rates if dilation_rates is not None else [2**i for i in range(n_layers)]


        self.network = nn.ModuleList() #define a list of nn.Modules for the ConvBlock

        self.network.append( #we need a first conv layer with 21 filters and no filters
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.n_filters,
                      kernel_size=conv1_kernel_size)
        )

        for i in range(1, self.n_layers + 1):
            in_channels = self.n_filters
            conv_layer = nn.Conv1d(in_channels=in_channels,
                                    out_channels=self.n_filters,
                                    kernel_size=self.kernel_size,
                                    dilation=2 ** i)
            self.network.append(conv_layer)


        self.network.append(self.pooling(self.pooling_size)) #pool after the block

    def forward(self, x):
        """
        x: (batch_size, in_channels, seq_len)
        """
        # Pass through the first convolutional layer
        x = self.network[0](x)  # First convolution layer

        # Save the output for residual connections
        residual = x 

        # Iterate through the subsequent layers, starting with the first dilated layer
        for i in range(1, self.n_layers + 1):
            # Get the current convolution layer (either dilated or last layer)
            conv_layer = self.network[i]
            
            # Forward through the convolution layer
            conv_output = conv_layer(x)

            # Only crop after the first layer (i.e., after the dilated layers)
            if i > 0:  # Ensure we crop only for dilated layers
                # Crop the residual to match the output size of the dilated conv
                x = self._crop(residual, conv_output)

                # Add the cropped input to the dilated convolution output (SKIP Connections)
                x = conv_output + x
                
            # Update residual for the next layer
            residual = x

          
        x = self.network[-1](x) # THIS IS THE POOLING LAYER

        return x

    
    def crop1d(x, target_len):
        """
        Crop the input tensor x to the target length in the 1st dimension.
        Assumes x is shaped (batch_size, channels, seq_len).
        """
        current_len = x.size(-1)
        crop_size = (current_len - target_len) // 2
        return x[:, :, crop_size:crop_size+target_len]




## Can try an RNN, etc
    
    