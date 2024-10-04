#Here is where the model architecture goes compare with bpnet_model.py in chrombpnet
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from utils.losses import multinomial_nll
from torch.nn import MSELoss
from ._base_modules import CNNModule, DilatedConvModule, Cropping1D, GlobalAvgPool1D, Flatten


class BPNetLightning(pl.LightningModule):
    def __init__(self, 
                 filters, 
                 n_dil_layers, 
                 conv1_kernel_size,
                 dilation_kernel_size,
                 profile_kernel_size, 
                 num_tasks, 
                 sequence_len, 
                 out_pred_len, 
                 learning_rate,
                 #counts_loss_weight,
                 #profile_loss_weight
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.mse_loss = MSELoss()
        
        #1) initial convolutional layer, use padding to preserve sequence length
        self.initial_conv = CNNModule(in_channels=4,
                                      out_channels=filters, 
                                      kernel_size=conv1_kernel_size, 
                                      padding=(conv1_kernel_size - 1) // 2)
        

        self.dilated_convs = nn.ModuleList()
        #self.crop_layers = nn.ModuleList()

        # Get the current length of the sequence for future cropping
        current_length = sequence_len - conv1_kernel_size + 1  # Adjust initial length after first conv

        for i in range(1, n_dil_layers + 1):
            dilation = 2 ** i
            conv = DilatedConvModule(in_channels=filters,
                                     out_channels=filters, 
                                     kernel_size=dilation_kernel_size, 
                                     dilation=dilation)
            self.dilated_convs.append(conv)


        #self.profile_conv = CNNModule(in_channels=filters,
        #                              out_channels=num_tasks, 
        #                              kernel_size=profile_kernel_size, 
        #                              padding=(profile_kernel_size - 1) // 2)
        
        #self.profile_crop = Cropping1D((sequence_len - (current_length + profile_kernel_size - 1) + 1) // 2)
        
        self.global_avg_pool = GlobalAvgPool1D()
        self.count_dense = nn.Linear(sequence_len, sequence_len)
        self.flatten = Flatten()

    def forward(self, x):
        #print(f"starting size:{x.shape}")
        x = self.initial_conv(x)
        #print(f"after first convolution:{x.shape}")
        
        #for i, (conv, crop_layer) in enumerate(zip(self.dilated_convs, self.crop_layers), 1):
        for i, conv in enumerate(self.dilated_convs, 1):
            # Apply dilated convolution (this does not reduce the length because we are padding)
            conv_x = conv(x)
            #print(f"after {i}th dilation:{conv_x.shape}")
            x = conv_x + x
      
        # We pool across the filters of the CNNs
        gap = self.global_avg_pool(x) 
        #print(f"after pool:{conv_x.shape}")

        # counts prediciton
        count_out = self.count_dense(gap)
        #print(f"count dimensions : {count_out.shape}")
        
        #return profile_out
        return count_out

    def training_step(self, train_batch, batch_idx): ## We call self.forward to go through the model and then calculate losses
        x, y = train_batch 
        y_hat_count = self(x)
        #loss_profile = self.profile_loss(y_hat_profile, y['profile']) # Get task specific losses
        loss_count = self.count_loss(y_hat_count, y)
        #loss = self.hparams.profile_loss_weight + loss_profile + self.hparams.counts_loss_weight * loss_count
        self.log('train_loss', loss_count, on_epoch=True,on_step=True, batch_size=y_hat_count.shape[0], sync_dist=True)
        return loss_count
    
    def validation_step(self, validation_batch, batch_idx): ## We call self.forward to go through the model and then calculate losses
        x, y = validation_batch 
        y_hat_count = self(x)
        #loss_profile = self.profile_loss(y_hat_profile, y['profile']) # Get task specific losses
        loss_count = self.count_loss(y_hat_count, y)
        #loss = self.hparams.profile_loss_weight + loss_profile + self.hparams.counts_loss_weight * loss_count
        self.log('validation_loss', loss_count, on_epoch=True,on_step=True, batch_size=y_hat_count.shape[0], sync_dist=True)
        return loss_count

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    #def profile_loss(self, predictions, targets):
    #    return nn.MSELoss()(predictions, targets) #stick with mse but we can add more further

    def count_loss(self, predictions, targets):
        #return multinomial_nll(predictions, targets) #for counts we gotta change the loss
        return self.mse_loss(predictions,targets)
    

    