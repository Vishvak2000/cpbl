#Here is where the model architecture goes compare with bpnet_model.py in chrombpnet
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError
from ._base_modules import CNNModule, DilatedConvModule, Cropping1D, GlobalAvgPool1D, Flatten


class BPNetLightning(pl.LightningModule):
    def __init__(self, 
                 filters, 
                 n_dil_layers, 
                 conv1_kernel_size, 
                 profile_kernel_size, 
                 num_tasks, 
                 sequence_len, 
                 out_pred_len, 
                 learning_rate):
        super().__init__()
        self.save_hyperparameters()
        
        #1) initial convolutional layer, use padding to preserve sequence length
        self.initial_conv = CNNModule(in_channels=4,
                                      out_channels=filters, 
                                      kernel_size=conv1_kernel_size, 
                                      padding=(conv1_kernel_size - 1) // 2)
        

        self.dilated_convs = nn.ModuleList()
        self.crop_layers = nn.ModuleList()

        # Get the current length of the sequence for future cropping
        current_length = sequence_len - conv1_kernel_size + 1  # Adjust initial length after first conv

        for i in range(1, n_dil_layers + 1):
            dilation = 2 ** i
            conv = DilatedConvModule(in_channels=filters,
                                     out_channels=filters, 
                                     kernel_size=3, 
                                     dilation=dilation)
            self.dilated_convs.append(conv)
            
            # Calculate cropping size to ensure the feature maps are the same size
            reduced_length = current_length - (2 * dilation)  # Calculate length reduction due to dilation and no padding
            crop_size = (current_length - reduced_length) // 2
            self.crop_layers.append(Cropping1D(crop_size))
            current_length = reduced_length # Series of croppings for each dilation, we do this so that we remove the epec
        
        self.profile_conv = CNNModule(in_channels=filters,
                                      out_channels=num_tasks, 
                                      kernel_size=profile_kernel_size, 
                                      padding=(profile_kernel_size - 1) // 2)
        
        self.profile_crop = Cropping1D((sequence_len - (current_length + profile_kernel_size - 1) + 1) // 2)
        self.global_avg_pool = GlobalAvgPool1D()
        self.count_dense = nn.Linear(filters, num_tasks)
        self.flatten = Flatten()

    def forward(self, x):
        x = self.initial_conv(x) # go through the initial convolution
        for conv, crop_layer in zip(self.dilated_convs, self.crop_layers):
            conv_x = conv(x)
            x = crop_layer(x)
            x = conv_x + x  # Element-wise addition (residual connection)
        
        prof = self.profile_conv(x)
        prof = self.profile_crop(prof)
        # profile prediction
        profile_out = self.flatten(prof) 
        

        # counts prediciton
        gap = self.global_avg_pool(x) 
        count_out = self.count_dense(gap)
        
        return profile_out, count_out

    def training_step(self, batch, batch_idx): ## work on this
        x, y = batch
        y_hat_profile, y_hat_count = self(x)
        loss_profile = self.profile_loss(y_hat_profile, y['profile'])
        loss_count = self.count_loss(y_hat_count, y['count'])
        loss = loss_profile + self.hparams.counts_loss_weight * loss_count
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def profile_loss(self, predictions, targets):
        return nn.MSELoss()(predictions, targets)

    def count_loss(self, predictions, targets):
        return nn.MSELoss()(predictions, targets)