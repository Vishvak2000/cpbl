import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from utils.losses import multinomial_nll
from torch.nn import MSELoss
from ._base_modules import CNNModule, DilatedConvModule, Cropping1D, GlobalAvgPool1D, Flatten
from utils.shape_utils import calculate_layer_output_length

class BPNetLightning(pl.LightningModule):
    def __init__(self, 
                 filters, 
                 n_dil_layers, 
                 conv1_kernel_size,
                 dilation_kernel_size,
                 sequence_len,
                 out_pred_len, 
                 learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.mse_loss = MSELoss()
        
        # Initial conv with no padding (valid padding)
        self.initial_conv = CNNModule(in_channels=4,
                                    out_channels=filters, 
                                    kernel_size=conv1_kernel_size, 
                                    padding=0)  # Changed to no padding
        
        self.dilated_convs = nn.ModuleList()
        self.crop_layers = nn.ModuleList()
        
        # Calculate the length after first convolution
        current_length = calculate_layer_output_length(
            sequence_len, 
            conv1_kernel_size, 
            padding=0
        )

        for i in range(1, n_dil_layers + 1):
            dilation = 2 ** i
            # Create dilated conv with no padding (valid padding)
            conv = DilatedConvModule(
                in_channels=filters,
                out_channels=filters, 
                kernel_size=dilation_kernel_size, 
                dilation=dilation,
                padding=0  # Changed to no padding
            )
            self.dilated_convs.append(conv)
            
            # Calculate new length after dilated conv
            conv_output_length = calculate_layer_output_length(
                current_length, 
                dilation_kernel_size, 
                dilation=dilation,
                padding=0
            )
            
            # Calculate crop size
            crop_size = (current_length - conv_output_length) // 2
    
            # Add a small adjustment if there is an off-by-one discrepancy
            if (current_length - conv_output_length) % 2 != 0:
                crop_size += 1

            self.crop_layers.append(Cropping1D(crop_size * 2))
            
            current_length = conv_output_length

        self.global_avg_pool = GlobalAvgPool1D()
        self.count_dense = nn.Linear(filters, out_pred_len)  # Modified to use filters as input

    def forward(self, x):
        x = self.initial_conv(x)
        
        for conv, crop_layer in zip(self.dilated_convs, self.crop_layers):
            # Apply dilated convolution
            conv_x = conv(x)
            # Crop the residual connection to match conv_x size
            x = crop_layer(x)
            # Add residual connection
            x = conv_x + x
        
        # Global average pooling across the sequence length
        x = self.global_avg_pool(x)
        
        # Final dense layer for count prediction
        count_out = nn.ReLU()(self.count_dense(x)) # add a relu here to force the outputs to be positive
        
        return count_out

    def forward_test(self, x):
            print(f"starting size:{x.shape}")
            x = self.initial_conv(x)
            print(f"after first convolution:{x.shape}")
            

            for i, (conv, crop_layer) in enumerate(zip(self.dilated_convs, self.crop_layers),1):
                # Apply dilated convolution (this does not reduce the length because we are padding)
                conv_x = conv(x)
                print(f"after {i}th dilation:{conv_x.shape}")
                x = crop_layer(x)
                # Add residual connection
                print(f"after {i}th crop:{x.shape}")
                x = conv_x + x
        
            # We pool across the filters of the CNNs
            gap = self.global_avg_pool(x) 
            print(f"after pool:{gap.shape}")

            # counts prediciton
            count_out = self.count_dense(gap)
            print(f"count dimensions : {count_out.shape}")
            count_out = nn.ReLU()(count_out)
            #return profile_out
            return count_out

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        y_hat_count = self(x)
        loss_count = self.count_loss(y_hat_count, y)
        self.log('train_loss', loss_count, on_epoch=True, on_step=True, batch_size=y_hat_count.shape[0], prog_bar = True)
        return loss_count
    
    def validation_step(self, validation_batch, batch_idx):
        x, y = validation_batch 
        y_hat_count = self(x)
        loss_count = self.count_loss(y_hat_count, y)
        self.log('validation_loss', loss_count, on_epoch=True, on_step=True, batch_size=y_hat_count.shape[0], prog_bar=True)
        return loss_count

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def count_loss(self, predictions, targets):
        return self.mse_loss(predictions, targets)