import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from utils import data_utils, one_hot
import multiprocessing

class ChromatinDataset(Dataset):
    def __init__(self, 
                 peak_regions, 
                 nonpeak_regions, 
                 genome_fasta, 
                 cts_bw_file, 
                 input_len, 
                 output_len, 
                 negative_sampling_ratio):
        self.input_len = input_len
        self.output_len = output_len
        self.negative_sampling_ratio = negative_sampling_ratio

        # Load data
        self.peak_df, self.nonpeak_df, self.genome, self.bw = data_utils.load_data(
            peak_regions, nonpeak_regions, genome_fasta, cts_bw_file
        )
        
        self.n_peaks = len(self.peak_df)
        self.n_nonpeaks = int(self.n_peaks * negative_sampling_ratio)
        
        print(f"Loaded {self.n_peaks} peak regions and {self.n_nonpeaks} non-peak regions")

    def __len__(self):
        return self.n_peaks + self.n_nonpeaks

    def __getitem__(self, idx):
        if idx < self.n_peaks:
            row = self.peak_df.iloc[idx]
            label = 1
        else:
            row = self.nonpeak_df.iloc[np.random.randint(len(self.nonpeak_df))]
            label = 0

        seq = data_utils.get_seq(self.genome, row['chr'], row['start'], row['end'], self.input_len)
        seq_one_hot = one_hot.dna_to_one_hot([seq])[0]  # Pass as list and take first element
        cts = data_utils.get_cts(self.bw, row['chr'], row['start'], row['end'], self.output_len)

        #print(f"seq_shape {seq_one_hot.shape}")
        #print(f"cts_shape {cts.shape}")

        return (torch.tensor(seq_one_hot, dtype=torch.float32).permute(1, 0),
                torch.tensor(cts, dtype=torch.float32))
                #torch.tensor(label, dtype=torch.float32))

    def split(self, train_chrs: list, valid_chrs: list, batch_size: int):
        train_indices = self.peak_df[self.peak_df['chr'].isin(train_chrs)].index.tolist()
        train_indices += list(range(self.n_peaks, len(self)))
        
        valid_indices = self.peak_df[self.peak_df['chr'].isin(valid_chrs)].index.tolist()

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        train_dataloader = DataLoader(self, sampler=train_sampler,
                                      num_workers=1,
                                      pin_memory=False, batch_size=batch_size)
        valid_dataloader = DataLoader(self, sampler=valid_sampler,
                                      num_workers=1,
                                      pin_memory=False, batch_size=batch_size)

        return train_dataloader, valid_dataloader

    def __del__(self):
        # Close the BigWig file when the dataset object is deleted
        if hasattr(self, 'bw'):
            self.bw.close()
        if hasattr(self, 'genome'):
            self.genome.close()