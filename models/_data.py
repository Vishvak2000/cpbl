import torch
from torch.utils.data import Dataset
import numpy as np
from utils import data_utils

class ChromatinDataset(Dataset):
    def __init__(self, 
                 peak_regions, 
                 nonpeak_regions, 
                 genome_fasta, 
                 cts_bw_file, 
                 inputlen, 
                 outputlen, 
                 max_jitter, 
                 negative_sampling_ratio):
        # Load data using the provided utility function
        self.peak_seqs, self.peak_cts, self.peak_coords, self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_coords = data_utils.load_data(
            peak_regions, nonpeak_regions, genome_fasta, cts_bw_file, inputlen, outputlen, max_jitter
        )
        
        # Sample nonpeak data
        self.sample_nonpeak_data(negative_sampling_ratio)
        
        # Concatenate peak and nonpeak data
        self.seqs = np.vstack([self.peak_seqs, self.sampled_nonpeak_seqs])
        self.cts = np.vstack([self.peak_cts, self.sampled_nonpeak_cts])
        self.coords = np.vstack([self.peak_coords, self.sampled_nonpeak_coords])
        
        # Convert to torch tensors
        self.seqs = torch.tensor(self.seqs, dtype=torch.float32)
        self.cts = torch.tensor(self.cts, dtype=torch.float32)

    def sample_nonpeak_data(self, ratio):
        num_samples = int(ratio * len(self.peak_seqs))
        indices = np.random.choice(len(self.nonpeak_seqs), size=num_samples, replace=False)
        self.sampled_nonpeak_seqs = self.nonpeak_seqs[indices]
        self.sampled_nonpeak_cts = self.nonpeak_cts[indices]
        self.sampled_nonpeak_coords = self.nonpeak_coords[indices]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.cts[idx]
