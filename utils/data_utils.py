import os
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
from . import one_hot

def process_bed(tsv_path):
    """Read a TSV file, select the first 3 columns, and rename them."""
    df = pd.read_csv(tsv_path, sep='\t', header=None)
    
    df_selected = df.iloc[:, :3]
    df_selected.columns = ['chr', 'start', 'end']
    print(f"Read in bed file of {df_selected.shape[0]} peaks")
    return df_selected

def get_seq(peaks_df, genome, input_len):
    """
    Fetch sequences from the genome using input_len centered on the regions.
    
    Args:
        peaks_df: DataFrame with 'chr', 'start', and 'end' columns.
        genome: Genome fasta loaded with pyfaidx.
        input_len: Length of the input sequence to fetch.
    
    Returns:
        One-hot encoded sequences centered on peaks.
    """
    vals = []
    for i, r in peaks_df.iterrows():
        # Calculate the center of the region
        center = (r['start'] + r['end']) // 2
        # Fetch the whole sequence based on input_len centered on peak
        sequence = str(genome[r['chr']][(center - input_len // 2):(center + input_len // 2)])
        vals.append(sequence)
    
    # Convert sequences to one-hot encoding
    return one_hot.dna_to_one_hot(vals)

def get_cts(peaks_df, bw, output_len):
    """
    Fetch counts from a bigwig bw file, centered at the middle of the peak region.
    
    Parameters:
    peaks_df (DataFrame): DataFrame with 'chr', 'start', and 'end' columns.
    bw (pyBigWig.BigWigFile): Open bigwig file for retrieving counts.
    output_len (int): Length of the counts to extract, centered on the peak.

    Returns:
    np.array: Array of counts centered on each peak.
    """
    vals = []
    for _, r in peaks_df.iterrows():
        start = int(r['start'])  # Ensure start is an integer
        end = int(r['end'])      # Ensure end is an integer
        center = (start + end) // 2  # Compute the center of the region

        # Fetch counts using output_len centered on the peak
        vals.append(np.nan_to_num(bw.values(r['chr'], 
                                            center - (output_len // 2),
                                            center + (output_len // 2))))
        
    return np.array(vals)

def get_coords(peaks_df, peaks_bool):
    """
    Fetch coordinates for the peaks.

    Args:
        peaks_df: DataFrame with 'chr', 'start', 'end' columns.
        peaks_bool: Boolean indicating if these are peaks (1) or non-peaks (0).
    
    Returns:
        Numpy array of coordinates centered on peaks.
    """
    vals = []
    for i, r in peaks_df.iterrows():
        # Calculate the center and return coordinates
        center = (r['start'] + r['end']) // 2
        vals.append([r['chr'], center, "f", peaks_bool])

    return np.array(vals)

def get_seq_cts_coords(peaks_df, genome, bw, input_len, output_len, peaks_bool):
    """
    Fetch sequences, counts, and coordinates for a given DataFrame.

    Args:
        peaks_df: DataFrame containing 'chr', 'start', and 'end' columns.
        genome: Genome fasta loaded with pyfaidx.
        bw: BigWig file opened with pyBigWig.
        input_len: Length of the input sequence to fetch.
        output_len: Length of the counts to fetch.
        peaks_bool: Boolean indicating if these are peaks (1) or non-peaks (0).
    
    Returns:
        Tuple containing sequences, counts, and coordinates.
    """
    seq = get_seq(peaks_df, genome, input_len)
    cts = get_cts(peaks_df, bw, output_len)
    coords = get_coords(peaks_df, peaks_bool)
    return seq, cts, coords

def load_data(bed_regions, nonpeak_regions, genome_fasta, cts_bw_file, input_len, output_len):
    """
    Load sequences and corresponding base-resolution counts for peaks and non-peaks.

    Args:
        bed_regions: Path to the peak regions BED file.
        nonpeak_regions: Path to the non-peak regions BED file.
        genome_fasta: Path to the genome fasta file.
        cts_bw_file: Path to the counts BigWig file.
        input_len: Length of the input sequence to fetch.
        output_len: Length of the counts to fetch.

    Returns:
        Tuple containing peak and non-peak sequences, counts, and coordinates.
    """
    cts_bw = pyBigWig.open(cts_bw_file)
    genome = pyfaidx.Fasta(genome_fasta)

    peak_regions_bed = process_bed(bed_regions).drop_duplicates()
    non_peak_regions_bed = process_bed(nonpeak_regions).drop_duplicates()

    # Initialize data for peaks and non-peaks
    train_peaks_seqs, train_peaks_cts, train_peaks_coords = None, None, None
    train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords = None, None, None

    # Load peak sequences, counts, and coordinates
    if bed_regions is not None:
        train_peaks_seqs, train_peaks_cts, train_peaks_coords = get_seq_cts_coords(
            peak_regions_bed, genome, cts_bw, input_len, output_len, peaks_bool=1
        )
    
    # Load non-peak sequences, counts, and coordinates
    if nonpeak_regions is not None:
        train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords = get_seq_cts_coords(
            non_peak_regions_bed, genome, cts_bw, input_len, output_len, peaks_bool=0
        )

    # Close BigWig and Genome Fasta files
    cts_bw.close()
    genome.close()

    return (
        train_peaks_seqs, train_peaks_cts, train_peaks_coords,
        train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords
    )
