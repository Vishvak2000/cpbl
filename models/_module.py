import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from utils.losses import FocusedMSELoss, MultinomialNLLLoss, CombinedLoss, spearman_corrcoef
from torch.nn import MSELoss
from ._base_modules import CNNModule, Flatten, DilatedConvModule, Cropping1D, GlobalAvgPool1D, AttentionPooling
from utils.shape_utils import calculate_layer_output_length
from torchmetrics.regression import KLDivergence, SpearmanCorrCoef, ExplainedVariance, CosineSimilarity, MeanAbsoluteError, R2Score, MeanSquaredLogError


class BPNetLightning(pl.LightningModule):
    def __init__(self, 
                 filters, 
                 n_dil_layers, 
                 conv1_kernel_size,
                 profile_kernel_size,
                 dilation_kernel_size,
                 sequence_len,
                 out_pred_len, # can add loss hyperparams here
                 learning_rate,
                 dropout_rate,
                 #multinomial_weight,
                 #mse_weight,
                 seq_focus_len,
                 avg_total_counts,
                 alpha,
                 flavor,
                 return_embeddings):
        super().__init__()
        self.save_hyperparameters()
        print(self.device)
        self.return_embeddings = return_embeddings
        self.eval_metrics = {
            "count": nn.ModuleDict({
                "mse" : MSELoss(),
                "mae" : MeanAbsoluteError(),
                "spearman" : SpearmanCorrCoef()
                }),

            "profile": nn.ModuleDict({
                #"focused_mse" : FocusedMSELoss(n=500),
                "mse" : MSELoss(),
                "kl_divergence" : KLDivergence(log_prob=False),
                "explained_variance" : ExplainedVariance(),
                "cosine_similarity" : CosineSimilarity(reduction="mean"),
                "mae" : MeanAbsoluteError(),
                "r2" : R2Score(),
                #"msle" : MeanSquaredLogError(),
                "spearman" : SpearmanCorrCoef(),
                "nll" : MultinomialNLLLoss()
                }),
        }
        
        self.loss = CombinedLoss(avg_total_counts=avg_total_counts,alpha = alpha,flavor=flavor) 
        
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
            #self.batch_norms.append(nn.BatchNorm1d(filters))
            
            # Calculate new length after dilated conv
            conv_output_length = calculate_layer_output_length(
                current_length, 
                dilation_kernel_size, 
                dilation=dilation,
                padding=0
            )

            crop_size = (current_length - conv_output_length) 
    
            # Add a small adjustment if there is an off-by-one discrepancy
            #if (current_length - conv_output_length) % 2 != 0:
            #    crop_size += 1

            #self.crop_layers.append(Cropping1D(crop_size * 2))
            self.crop_layers.append(Cropping1D(crop_size))
            
            current_length = conv_output_length
        
        self.profile_conv = CNNModule(in_channels=filters,
                                    out_channels=1, 
                                    kernel_size=profile_kernel_size, 
                                    padding=0)
        
        conv_output_length = calculate_layer_output_length(
                current_length, 
                profile_kernel_size, 
                dilation=1,
                padding=0
            )
        
        crop_size = (conv_output_length - out_pred_len) // 2
        if (current_length - out_pred_len) % 2 != 0:
                crop_size += 1

        
        self.profile_crop = Cropping1D(crop_size*2)
        self.profile_flatten = Flatten()
        

        ## Now for counts prediction

        self.global_avg_pool = GlobalAvgPool1D()
        
        if dropout_rate != 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.count_dense = nn.Linear(filters, 1)
    
    def forward(self, x, return_embeddings=False):
        self.return_embeddings = return_embeddings
        embeddings = {}

        x = self.initial_conv(x)
        embeddings['initial_conv'] = x.clone()

        for conv, crop_layer in zip(self.dilated_convs, self.crop_layers):
            conv_x = conv(x)
            x = crop_layer(x)
            x = conv_x + x
        
        embeddings['after_dilated_convs'] = x.clone()

        ### profile prediction
        prof = self.profile_conv(x)
        prof = self.profile_crop(prof)
        prof = self.profile_flatten(prof)

        ## count predicition
        x = self.global_avg_pool(x)
        
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        
        count_out = (self.count_dense(x))
        embeddings['global_pooled'] = x.clone()

        if return_embeddings:
            return [count_out, prof, embeddings]
        return [count_out, prof]

    

   
    def forward_test(self, x):
        """
        Detailed forward pass for testing and debugging
        
        Returns:
            tuple: (profile_predictions, count_predictions)
        """
        print(f"starting size:{x.shape}")
        x = self.initial_conv(x)
        print(f"after first convolution:{x.shape}")
        
        for i, (conv, crop_layer) in enumerate(zip(self.dilated_convs, self.crop_layers), 1):
            conv_x = conv(x)
            print(f"after {i}th dilation:{conv_x.shape}")
            x = crop_layer(x)
            print(f"after {i}th crop:{x.shape}")
            x = conv_x + x
        
        # Profile prediction
        prof = self.profile_conv(x)
        prof = self.profile_crop(prof)
        prof = self.profile_flatten(prof)
        
        # Count prediction
        x = self.global_avg_pool(x)
        
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        
        count_out = self.count_dense(x)
        
        print(f"Profile prediction shape: {prof.shape}")
        print(f"Count prediction shape: {count_out.shape}")
        
        return [count_out, prof]

    def calculate_metrics(self, y_hat, y):
        """
        Separately calculates metrics for count prediction and profile prediction.
        Args:
            y_hat (list): List containing [count_pred, profile_pred].
            y (list): List containing [count_true, profile_true].
        Returns:
            dict: Dictionary of calculated metrics for each output type.
        """
        results = {}
        
        # Parse outputs
        count_pred, profile_pred = y_hat
        count_true, profile_true = y
        
        # Count prediction metrics
        for metric_name, metric_fn in self.eval_metrics["count"].items():
            if metric_name == "spearman":
                self.eval_metrics["count"]["spearman"].num_outputs = count_pred.shape[0]
                self.eval_metrics["profile"]["spearman"].num_outputs = count_pred.shape[0]
                #print(count_pred.shape[0])
                results[f"count_{metric_name}"] = torch.mean(metric_fn(count_pred.squeeze(), count_true.squeeze())).item()
                #results[f"count_{metric_name}"] = torch.mean(spearman_corrcoef(count_pred.T, count_true.T)).item()

            else:
                results[f"count_{metric_name}"] = metric_fn(count_pred, count_true).item()
        
        # Profile prediction metrics
        for metric_name, metric_fn in self.eval_metrics["profile"].items():
            if metric_name == "spearman":
                self.eval_metrics["profile"]["spearman"].num_outputs = profile_pred.shape[0]
                results[f"profile_{metric_name}"] = torch.mean(metric_fn(profile_pred.T, profile_true.T)).item()
                #results[f"count_{metric_name}"] = torch.mean(spearman_corrcoef(profile_pred.T, profile_true.T)).item()

            elif metric_name == "kl_divergence":
                eps = 1e-8
                y_hat_prob = (profile_pred + eps) / (profile_pred.sum() + eps * profile_pred.shape[-1])
                y_prob = (profile_true + eps) / (profile_true.sum() + eps * profile_true.shape[-1])
                results[f"profile_{metric_name}"] = metric_fn(y_hat_prob, y_prob).item()
            elif metric_name == "r2":
                results[f"profile_{metric_name}"] = metric_fn(profile_pred.reshape(-1), profile_true.reshape(-1)).item()
            else:
                results[f"profile_{metric_name}"] = metric_fn(profile_pred, profile_true).item()
        
        return results
    
    
    def log_metrics(self, y_hat, y, prefix):
        """
        Logs metrics separately for count prediction and profile prediction.
        Args:
            y_hat (list): List containing [count_pred, profile_pred].
            y (list): List containing [count_true, profile_true].
            prefix (str): Prefix for logging metrics (e.g., 'train' or 'val').
        Returns:
            torch.Tensor: Combined loss for both outputs.
        """
        # Calculate metrics
        metrics = self.calculate_metrics(y_hat, y)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"{prefix}_{metric_name}", metric_value, on_step=False, on_epoch=True, logger=True)
        
        # Calculate combined loss
        loss = self.loss(y_hat,y)
        self.log(f"{prefix}_combined_loss", loss, on_step=False, on_epoch=True, logger=True)
        
        return loss


    def training_step(self, train_batch, batch_idx):
        if self.return_embeddings:
            raise ValueError("Embedding extraction should not be used during training.")
        x, y = train_batch  # y is a list: [count_true, profile_true]
        y_hat = self(x)  # Forward pass; y_hat is [count_pred, profile_pred]
        loss = self.log_metrics(y_hat, y, "train")
        torch.cuda.empty_cache()
        return loss

    @torch.no_grad()
    def validation_step(self, validation_batch, batch_idx):
        x, y = validation_batch  # y is a list: [count_true, profile_true]
        y_hat = self(x)  # Forward pass; y_hat is [count_pred, profile_pred]
        loss = self.log_metrics(y_hat, y, "val")
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": f"val_combined_loss",  # Replace with the actual metric you want to monitor
                "interval": "epoch",
                "frequency": 3,
            },
        }


    @torch.no_grad()
    def predict(self, x, return_embeddings=False):
        """
        Perform inference with the model.
        
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, 4, sequence_len).
            return_embeddings (bool): If True, also return intermediate embeddings.

        Returns:
            tuple: (count_predictions, profile_predictions, embeddings [optional])
        """
        self.eval()  # Ensure the model is in evaluation mode
        output = self(x, return_embeddings=return_embeddings)

        if return_embeddings:
            count_pred, profile_pred, embeddings = output
            return count_pred, profile_pred, embeddings
        else:
            count_pred, profile_pred = output
            return count_pred, profile_pred

    def on_fit_start(self):
        """Move all metrics to the appropriate device."""
        print(f"Model is on {self.device}, moving metrics to {self.device}")
        for metric_group in self.eval_metrics.values():
            for metric_name, metric_fn in metric_group.items():
                metric_group[metric_name] = metric_fn.to(self.device)