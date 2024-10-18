import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from utils.losses import WeightedMSELoss, FocusedMSELoss
from torch.nn import MSELoss
from ._base_modules import CNNModule, DilatedConvModule, Cropping1D, GlobalAvgPool1D, Flatten
from utils.shape_utils import calculate_layer_output_length
from torchmetrics.regression import KLDivergence, ExplainedVariance, CosineSimilarity, MeanAbsoluteError, R2Score


class BPNetLightning(pl.LightningModule):
    def __init__(self, 
                 filters, 
                 n_dil_layers, 
                 conv1_kernel_size,
                 dilation_kernel_size,
                 sequence_len,
                 out_pred_len, # can add loss hyperparams here
                 learning_rate,
                 dropout_rate,
                 seq_focus_len):
        super().__init__()
        self.save_hyperparameters()

     
        self.eval_metrics = nn.ModuleDict({
            "weighted_mse" : WeightedMSELoss(sequence_length=out_pred_len,center_size=seq_focus_len),
            "focused_mse" : FocusedMSELoss(n=500),
            "mse" : MSELoss(),
            "kl_divergence" : KLDivergence(log_prob=False),
            "explained_variance" : ExplainedVariance(),
            "cosine_similarity" : CosineSimilarity(),
            "mae" : MeanAbsoluteError(),
            "r2" : R2Score()
        })
        
        
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

        if dropout_rate!=0.0:
            self.dropout = nn.Dropout(dropout_rate)
    

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

        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        
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

    def calculate_metrics(self, y_hat, y):
        results = {}
        for metric_name, metric_fn in self.eval_metrics.items():
            if metric_name == "kl_divergence":
                results[metric_name] = metric_fn((y_hat/y_hat.sum()), (y/y.sum()))
            if metric_name == "r2":
                results[metric_name] = metric_fn((y_hat.view(-1)), (y.view(-1)))
            else:
                results[metric_name] = metric_fn(y_hat, y)
        
        return results
    
    def log_metrics(self, y_hat, y, prefix):
        metrics = self.calculate_metrics(y_hat, y)

        for metric_name, metric_value in metrics.items():
            self.log(f"{prefix}_{metric_name}", metric_value, on_step=False, on_epoch=True, logger=True)
        
        #try separately calculating weighted mse - maybe thats the issue?
        weighted_mse = self.eval_metrics["weighted_mse"](y_hat, y)
        return weighted_mse  


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        y_hat_count = self(x)
        loss = self.log_metrics(y_hat_count,y,"train")
        #self.log('train_loss', loss_count, on_epoch=True, on_step=True, batch_size=y_hat_count.shape[0], prog_bar = True)
        return loss
    
    def validation_step(self, validation_batch, batch_idx):
        x, y = validation_batch 
        y_hat_count = self(x)
        #loss_count = self.count_loss(y_hat_count, y)
        loss = self.log_metrics(y_hat_count,y,"val")
        #self.log('train_loss', loss_count, on_epoch=True, on_step=True, batch_size=y_hat_count.shape[0], prog_bar = True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def count_loss(self, predictions, targets):
        return self.mse_loss(predictions, targets)