import os
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
from . import one_hot

def process_bed(tsv_path):
    """Read a TSV file, select the first 3 columns, and rename them."""
    df = pd.read_csv(tsv_path, sep='\t', header=None, usecols=[0, 1, 2])
    df.columns = ['chr', 'start', 'end']
    print(f"Read in bed file of {df.shape[0]} regions")
    return df

def get_seq(genome, chrom, start, end, input_len):
    """Fetch a sequence from the genome ensuring fixed length."""
    center = (start + end) // 2
    seq_start = max(0, center - input_len // 2)
    seq_end = seq_start + input_len
    
    # Ensure the sequence length is always equal to input_len
    sequence = str(genome[chrom][seq_start:seq_end])
    if len(sequence) < input_len:
        # Pad the sequence if it's shorter than input_len (in case of edge cases at the chromosome ends)
        sequence = sequence + 'N' * (input_len - len(sequence))
    elif len(sequence) > input_len:
        # Truncate if necessary
        sequence = sequence[:input_len]
    
    return sequence
    

def get_cts(bw, chrom, start, end, output_len):
    """Fetch counts from a bigwig file."""
    center = (start + end) // 2
    cts_start = center - output_len // 2
    cts_end = cts_start + output_len
    return np.nan_to_num(bw.values(chrom, cts_start, cts_end))


def load_data(bed_regions, nonpeak_regions, genome_fasta, cts_bw_file):
    """Load peak and non-peak regions."""
    peak_df = process_bed(bed_regions)
    nonpeak_df = process_bed(nonpeak_regions)
    
    genome = pyfaidx.Fasta(genome_fasta)
    bw = pyBigWig.open(cts_bw_file)
    
    return peak_df, nonpeak_df, genome, bw



