import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from ._data import ChromatinDataset
from ._module import BPNetLightning 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from typing import Optional, Union
from utils.shape_utils import calculate_required_input_length
from pytorch_lightning.loggers import WandbLogger



import os

class CBPLTrainer:
    def __init__(self,config):
        self.config = config

        # Load the dataset
        self.dataset = ChromatinDataset(
            peak_regions=config["peak_regions"],
            nonpeak_regions=config["nonpeak_regions"],
            genome_fasta=config["genome_fasta"],
            cts_bw_file=config["cts_bw_file"],
            input_len=config["input_seq_len"],
            output_len=config["out_pred_len"],
            negative_sampling_ratio=config["negative_sampling_ratio"]
        )

        # Create data loaders
        self.train_dataloader, self.valid_dataloader = self.dataset.split(
            #train_size=config["train_size"], 
            batch_size=config["batch_size"],
            train_chrs=config["train_chrs"],
            valid_chrs=config["valid_chrs"]
            )

        # Instantiate the model
        self.model = BPNetLightning(
            filters=config["filters"],
            n_dil_layers=config["n_dil_layers"],
            conv1_kernel_size=config["conv1_kernel_size"],
            dilation_kernel_size = config["dilation_kernel_size"],
            sequence_len=config["input_seq_len"],
            out_pred_len=config["out_pred_len"],
            learning_rate=config["learning_rate"],
            dropout_rate = config["dropout_rate"],
            seq_focus_len=config["seq_focus_len"],
            loss=config["loss"]
        )
    

        required_input_length = calculate_required_input_length(self.config['out_pred_len'],config)
        print(f"Given config['out_pred_len'] = {self.config['out_pred_len']}, we need input_len = {required_input_length} \n")
        print(f"Current sequence length is {self.config['input_seq_len']}")
        
    def fit(self, max_epochs: int = 50,
            batch_size: int = 128,
            early_stopping_patience: int = 5,
            #train_size: Optional[float] = 0.9,
            check_val_every_n_epoch: Optional[Union[int, float]] = 1,
            save_path: Optional[str] = None,
            logger_out: Optional[WandbLogger] = None,
            gpus = None):
        
        es_callback = EarlyStopping(monitor='val_weighted_mse', patience=early_stopping_patience, verbose=True, mode='min')

        self.trainer = pl.Trainer(max_epochs=max_epochs,
                                  #accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                                  #devices = gpus,
                                  #distributed_backend='ddp',
                                  logger=logger_out,
                                  check_val_every_n_epoch=check_val_every_n_epoch,
                                  enable_progress_bar=True,
                                  default_root_dir=save_path,
                                  callbacks=[es_callback])
        self.trainer.fit(self.model, self.train_dataloader, self.valid_dataloader)

    def save_model(self, save_dir):
        # Save the model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, 'chrombpnet_model.ckpt')
        self.model.save_model(model_path)
        print(f"Model saved at {model_path}")
