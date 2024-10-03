import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from utils import data_utils
import logging
import multiprocessing

class ChromatinDataset(Dataset):
    def __init__(self, 
                 peak_regions, 
                 nonpeak_regions, 
                 genome_fasta, 
                 cts_bw_file, 
                 #inputlen, 
                 #outputlen, 
                 #max_jitter, 
                 negative_sampling_ratio):
        # Load data using the provided utility function
        self.peak_seqs, self.peak_cts, self.peak_coords, self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_coords = data_utils.load_data(
            peak_regions, nonpeak_regions, genome_fasta, cts_bw_file
        )

        
        # Sample nonpeak data
        self.sample_nonpeak_data(negative_sampling_ratio)
        
        print(f"Successfully loaded in data with {len(self.peak_seqs)} positive and {len(self.sampled_nonpeak_seqs)} nonpeak regions!")

        # Concatenate peak and nonpeak data
        self.seqs = np.vstack([self.peak_seqs, self.sampled_nonpeak_seqs])
        self.cts = np.vstack([self.peak_cts, self.sampled_nonpeak_cts])
        self.coords = np.vstack([self.peak_coords, self.sampled_nonpeak_coords])
        
        # Convert to torch tensors
        self.seqs = torch.tensor(self.seqs, dtype=torch.float32)
        self.cts = torch.tensor(self.cts, dtype=torch.float32)

    def sample_nonpeak_data(self, ratio):
        num_samples = int(ratio * len(self.peak_seqs))
        num_samples = min(num_samples, len(self.nonpeak_seqs)) # for smaller datasets
        indices = np.random.choice(len(self.nonpeak_seqs), size=num_samples, replace=False)
        self.sampled_nonpeak_seqs = self.nonpeak_seqs[indices]
        self.sampled_nonpeak_cts = self.nonpeak_cts[indices]
        self.sampled_nonpeak_coords = self.nonpeak_coords[indices]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.cts[idx]

    def split(self, train_size: float, batch_size: int):
        n_samples = len(self.seqs)

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        train_indices = indices[:int(train_size * n_samples)]
        valid_indices = indices[int(train_size * n_samples):]

        train_dataloader = DataLoader(self, sampler=SubsetRandomSampler(train_indices),
                                      num_workers=max(1, min(6, multiprocessing.cpu_count() // 2)), pin_memory=False,
                                      batch_size=batch_size)
        valid_dataloader = DataLoader(self, sampler=SubsetRandomSampler(valid_indices),
                                      num_workers=max(1, min(6, multiprocessing.cpu_count() // 2)), pin_memory=False,
                                      batch_size=batch_size)

        return train_dataloader, valid_dataloader