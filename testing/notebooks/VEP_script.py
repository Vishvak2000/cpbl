import h5py
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from utils.data_utils import get_seq, get_cts
from utils.one_hot import dna_to_one_hot
import pyBigWig
import pyfaidx
import sys
import os
import pickle

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join('../..')))

from models._data import ChromatinDataset
from models._model import CBPLTrainer


def prepare_variant_sequences(variants_df, genome, input_len=2114):
    """
    Prepare reference and alternate sequences for variant analysis.
    """
    
    # Initialize lists to store sequences
    ref_sequences = []
    alt_sequences = []
    mismatch_count = 0  # Initialize mismatch counter
    
    # Process each variant
    for _, row in variants_df.iterrows():
        # Extract chromosome, position, and alleles
        chrom = str(row['Chromosome'])
        pos = row['SNP_position']
        ref_allele = row['Ref_allele']
        alt_allele = row['Alt_allele']
        
        # Get reference sequence centered on the SNP position
        ref_seq = get_seq(genome, chrom, pos, pos, input_len)
        
        # Center position in the sequence
        center_pos = input_len // 2 -1
        
        # Validate that the reference allele matches the center position
        if ref_seq[center_pos] != ref_allele:
            mismatch_count += 1  # Increment mismatch counter
            print(f"Warning: Mismatch at {chrom}:{pos}. "
                  f"Expected {ref_allele}, found {ref_seq[center_pos]}")
        
        # Create reference and alternate sequences
        ref_seq_list = list(ref_seq)
        ref_seq_list[center_pos] = ref_allele
        ref_sequences.append(''.join(ref_seq_list))
        
        alt_seq_list = list(ref_seq)
        alt_seq_list[center_pos] = alt_allele
        alt_sequences.append(''.join(alt_seq_list))
    
    # Print the total number of mismatches
    print(f"Total mismatches: {mismatch_count}")
    
    return {
        'variants_df': variants_df,
        'ref_sequences': ref_sequences,
        'alt_sequences': alt_sequences,
    }




train_chrs = ["chr2",
    "chr4",
    "chr5",
    "chr7",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr21",
    "chr22",
    "chrX",
    "chrY"]


config = {
    "peak_regions": "/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/pd_data/Microglia_peak_set_2.bed",
    "nonpeak_regions": "/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/own_data/test.chr1.negatives.adjusted.bed",
    "genome_fasta": "/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/data/downloads/hg38.fa",
    "cts_bw_file": "/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/pd_data/nd_Microglia_merge.bw",
    "negative_sampling_ratio": 0,
    "train_size": 0.6,
    "batch_size": 32,
    "filters": 512,
    "n_dil_layers": 8,
    "conv1_kernel_size": 21,
    "profile_kernel_size": 71,
    "dilation_kernel_size": 3,
    "input_seq_len": 2114,
    "out_pred_len": 1000,
    "dropout_rate": 0.0,
    "learning_rate": 0.001,
    "train_chrs": train_chrs,
    "valid_chrs": ["chr1"],
    "seq_focus_len": 500,
    "use_cpu": False,
    "alpha" : 1,
    "checkpoint_path": None,
    "flavor" : None,
    "project": "cbpl_new_microglia",
    "return_embeddings" : True,
    "jitter" : False,
    "jitter_scale" : 0.1
}


trainer =  CBPLTrainer(config,checkpoint_path='/wynton/home/corces/vishvak/pytorch_cbp/testing/cbpl_new_microglia/p4d4disj/checkpoints/cbpl-epoch=09-val_loss=0.00.ckpt')
genome = pyfaidx.Fasta("/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/data/downloads/hg38.fa")
variants = pd.read_csv("Microglia_RASQUAL_results_FDR_0.05.txt",sep='\t')
variants["type_ref"] = variants["Ref_allele"].apply(len)
variants["type_alt"] = variants["Alt_allele"].apply(len)
variants = variants[(variants["type_ref"] == 1) &( variants["type_alt"] == 1)]


sequence_dict = prepare_variant_sequences(variants,genome)

ref_sequences = torch.tensor(dna_to_one_hot(sequence_dict["ref_sequences"]), dtype=torch.float32).permute(0,2,1)
alt_sequences = torch.tensor(dna_to_one_hot(sequence_dict["alt_sequences"]), dtype=torch.float32).permute(0,2,1)

ref_predictions = trainer.model.predict(ref_sequences.to('cuda'),return_embeddings=True)
alt_predictions = trainer.model.predict(alt_sequences.to('cuda'),return_embeddings=True)


with open('ref_predictions.pckl', 'wb') as f:
    pickle.dump(ref_predictions, f)

with open('alt_predictions.pckl', 'wb') as f:
    pickle.dump(ref_predictions, f)