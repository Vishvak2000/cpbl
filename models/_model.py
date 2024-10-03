import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ._data import ChromatinDataset
from ._module import BPNetLightning 
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
            negative_sampling_ratio=config["negative_sampling_ratio"]
        )

        # Create data loaders
        self.train_dataloader, self.valid_dataloader = self.dataset.split(
            train_size=config["train_size"], 
            batch_size=config["batch_size"]
        )

        # Instantiate the model
        self.model = BPNetLightning(
            filters=config["filters"],
            n_dil_layers=config["n_dil_layers"],
            conv1_kernel_size=config["conv1_kernel_size"],
            profile_kernel_size=config["profile_kernel_size"],
            num_tasks=config["num_tasks"],
            sequence_len=config["sequence_len"],
            out_pred_len=config["out_pred_len"],
            learning_rate=config["learning_rate"]
        )

        
    def fit(self, epochs):
        # Set up the PyTorch Lightning trainer
        trainer = pl.Trainer(max_epochs=epochs,
                             enable_progress_bar=True,
                             )
        
        # Train the model
        trainer.fit(self.model, self.train_dataloader, self.valid_dataloader)
        
    def save_model(self, save_dir):
        # Save the model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, 'chrombpnet_model.ckpt')
        self.model.save_model(model_path)
        print(f"Model saved at {model_path}")
