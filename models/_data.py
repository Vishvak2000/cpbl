import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from utils import data_utils, one_hot
import multiprocessing
import pyBigWig
import pyfaidx

class ChromatinDataset(Dataset):
    def __init__(self, 
                 peak_regions, 
                 nonpeak_regions, 
                 genome_fasta, 
                 cts_bw_file, 
                 input_len, 
                 output_len, 
                 negative_sampling_ratio,
                 jitter: bool = False,  # New parameter with default False
                 jitter_scale: float = 0.1):
        self.jitter = jitter
        self.jitter_scale = jitter_scale
        self.input_len = input_len
        self.output_len = output_len
        self.negative_sampling_ratio = negative_sampling_ratio
        self.genome_fasta = genome_fasta
        self.cts_bw_file = cts_bw_file

        # Load data
        self.peak_df, self.nonpeak_df = data_utils.load_data(
            peak_regions, nonpeak_regions
        )
        
        self.n_peaks = len(self.peak_df)
        self.n_nonpeaks = int(self.n_peaks * negative_sampling_ratio)
        
        print(f"Loaded {self.n_peaks} peak regions and {self.n_nonpeaks} non-peak regions")

        # Initialize these as None, they will be created in each worker
        self.genome = None
        self.bw = None

    def __len__(self):
        return self.n_peaks + self.n_nonpeaks

    def _init_genome_and_bw(self):
        if self.genome is None:
            self.genome = pyfaidx.Fasta(self.genome_fasta)
        if self.bw is None:
            self.bw = pyBigWig.open(self.cts_bw_file)

    def __getitem__(self, idx):
        self._init_genome_and_bw()

        if idx < self.n_peaks:
            row = self.peak_df.iloc[idx]
            label = 1
        else:
            row = self.nonpeak_df.iloc[np.random.randint(len(self.nonpeak_df))]
            label = 0

        seq = data_utils.get_seq(self.genome, row['chr'], row['start'], row['end'], self.input_len)
        
        seq_one_hot = one_hot.dna_to_one_hot([seq])[0]
        cts = data_utils.get_cts(self.bw, row['chr'], row['start'], row['end'], self.output_len)
    
        # Conditional jitter
        if self.jitter:
            cts_processed = cts + np.random.normal(0, self.jitter_scale * cts.std(), size=cts.shape)
            cts_processed = np.maximum(cts_processed, 0)  # Ensure non-negative
        else:
            cts_processed = cts
        
        log_cts = np.log(1 + cts_processed.sum(-1, keepdims=True))
        log_cts_tensor = torch.tensor(log_cts, dtype=torch.float32)
        
        return (torch.tensor(seq_one_hot, dtype=torch.float32).permute(1, 0),
                [log_cts_tensor, torch.tensor(cts_processed, dtype=torch.float32)])

    def split(self, train_chrs: list, valid_chrs: list, batch_size: int):
        train_indices = self.peak_df[self.peak_df['chr'].isin(train_chrs)].index.tolist()
        train_indices += list(range(self.n_peaks, len(self)))
        
        valid_indices = self.peak_df[self.peak_df['chr'].isin(valid_chrs)].index.tolist()

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        train_dataloader = DataLoader(self, sampler=train_sampler,
                                      num_workers=3, # set explicitly to 3 currently
                                      pin_memory=True, batch_size=batch_size)
        valid_dataloader = DataLoader(self, sampler=valid_sampler,
                                      num_workers=3,
                                      pin_memory=True, batch_size=batch_size)

        return train_dataloader, valid_dataloader

    def __del__(self):
        if hasattr(self, 'bw') and self.bw is not None:
            self.bw.close()
        if hasattr(self, 'genome') and self.genome is not None:
            self.genome.close()