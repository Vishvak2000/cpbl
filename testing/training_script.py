import torch
import torch.nn as nn
import sys
import os

# Set the number of threads to use (30% of available CPU cores)
total_cores = os.cpu_count()
num_cores_to_use = max(1, int(total_cores * 0.3))  # Ensure at least one core is used
torch.set_num_threads(num_cores_to_use)

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join('..')))

from models._model import CBPLTrainer


config = {
    "peak_regions":"/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/own_data/test.chr1.adjusted.bed",
    "nonpeak_regions":"/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/own_data/test.chr1.negatives.adjusted.bed",
    "genome_fasta":"/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/own_data/chr1.fa",
    "cts_bw_file":"/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/own_data/test.chr1.bw",
    "negative_sampling_ratio": 0.8,
    "train_size": 0.7,
    "batch_size": 32,
    "filters": 64,
    "n_dil_layers": 3,
    "conv1_kernel_size": 7,
    "dilation_kernel_size" : 3,
    "profile_kernel_size": 5,
    "sequence_len": 1000,
    "out_pred_len": 1000,
    "learning_rate": 0.001,
}


trainer =  CBPLTrainer(config)

trainer.model

trainer.fit()