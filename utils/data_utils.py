import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
from . import one_hot
from collections import Counter


def process_bed(tsv_path):
    """Read a TSV file, select the first 3 columns, and rename them."""
    df = pd.read_csv(tsv_path, sep='\t', header=None)
    df_selected = df.iloc[:, :3]
    df_selected.columns = ['chr', 'start', 'end']
    print(f"Read in bed file of {df_selected.shape[0]} peaks")
    return df_selected


def get_seq(peaks_df,genome):
    """
    Here we're assuming the sequences are already centered
    """
    vals = []

    for i, r in peaks_df.iterrows():
        sequence = str(genome[r['chr']][(r['start']):(r['end'])])
        vals.append(sequence)
    
    # Get value_counts() for each element in the list
    lengths = peaks_df["end"] - peaks_df["start"]
    print(f"Peak length = {lengths.value_counts()}")
    return one_hot.dna_to_one_hot(vals)


def get_cts(peaks_df,bw):
    """
    Here we are assuming it is already centered at the peak
    """
    vals = [] 
    for i, r in peaks_df.iterrows():
        vals.append(np.nan_to_num(bw.values(r['chr'], 
                                            r['start'],
                                            r['end'])))
        
    return np.array(vals)

def get_coords(peaks_df,peaks_bool):
    """
    Assuming we are already centered
    """
    vals = []
    for i, r in peaks_df.iterrows():
        vals.append([r['chr'], r['start']+r['end'] // 2, "f", peaks_bool])

    return np.array(vals)

def get_seq_cts_coords(peaks_df, genome, bw, peaks_bool):

    seq = get_seq(peaks_df, genome)
    cts = get_cts(peaks_df, bw)
    coords = get_coords(peaks_df, peaks_bool)
    return seq, cts, coords


def load_data(bed_regions, nonpeak_regions, genome_fasta, cts_bw_file):
    """
    Here we assume the bed files are already centered on the summit - we also don't care about jitter to keep it simple

    Load sequences and corresponding base resolution counts for training, 
    validation regions in peaks and nonpeaks (2 x 2 x 2 = 8 matrices).

    For training peaks/nonpeaks, values for inputlen + 2*max_jitter and outputlen + 2*max_jitter 
    are returned centered at peak summit. This allows for jittering examples by randomly
    cropping. Data of width inputlen/outputlen is returned for validation
    data.

    If outliers is not None, removes training examples with counts > outlier%ile
    """

    cts_bw = pyBigWig.open(cts_bw_file)
    genome = pyfaidx.Fasta(genome_fasta)

    peak_regions_bed = process_bed(bed_regions).drop_duplicates()
    non_peak_regions_bed = process_bed(nonpeak_regions).drop_duplicates()

    train_peaks_seqs=None
    train_peaks_cts=None
    train_peaks_coords=None
    train_nonpeaks_seqs=None
    train_nonpeaks_cts=None
    train_nonpeaks_coords=None

    if bed_regions is not None:
        train_peaks_seqs, train_peaks_cts, train_peaks_coords = get_seq_cts_coords(peak_regions_bed,
                                              genome,
                                              cts_bw,
                                              peaks_bool=1)
    
    if nonpeak_regions is not None:
        train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords = get_seq_cts_coords(non_peak_regions_bed,
                                              genome,
                                              cts_bw,
                                              peaks_bool=0)



    cts_bw.close()
    genome.close()

    return (train_peaks_seqs, train_peaks_cts, train_peaks_coords,
            train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords)

################### ############################################THIS IS ALL OLD STUFF ##################################################################
####################################################################################################################################
####################################################################################################################################
##########################################################################################################################################################
def get_seq_old(peaks_df, genome):
    """
    Same as get_cts, but fetches sequence from a given genome.
    """
    vals = []

    for i, r in peaks_df.iterrows():
        sequence = str(genome[r['chr']][(r['start']+r['summit'] - width//2):(r['start'] + r['summit'] + width//2)])
        vals.append(sequence)

    print(vals)
    return one_hot.dna_to_one_hot(vals)

def get_cts_old(peaks_df, bw, width):
    """
    Fetches values from a bigwig bw, given a df with minimally
    chr, start and summit columns. Summit is relative to start.
    Retrieves values of specified width centered at summit.

    "cts" = per base counts across a region
    """
    vals = []
    for i, r in peaks_df.iterrows():
        vals.append(np.nan_to_num(bw.values(r['chr'], 
                                            r['start'] + r['summit'] - width//2,
                                            r['start'] + r['summit'] + width//2)))
        
    return np.array(vals)

def get_coords_old(peaks_df, peaks_bool):
    """
    Fetch the co-ordinates of the regions in bed file
    returns a list of tuples with (chrom, summit)
    """
    vals = []
    for i, r in peaks_df.iterrows():
        vals.append([r['chr'], r['start']+r['summit'], "f", peaks_bool])

    return np.array(vals)

def get_seq_cts_coords_old(peaks_df, genome, bw, input_width, output_width, peaks_bool):

    seq = get_seq(peaks_df, genome, input_width)
    cts = get_cts(peaks_df, bw, output_width)
    coords = get_coords(peaks_df, peaks_bool)
    return seq, cts, coords

def load_data_old(bed_regions, nonpeak_regions, genome_fasta, cts_bw_file, inputlen, outputlen, max_jitter):
    """
    Load sequences and corresponding base resolution counts for training, 
    validation regions in peaks and nonpeaks (2 x 2 x 2 = 8 matrices).

    For training peaks/nonpeaks, values for inputlen + 2*max_jitter and outputlen + 2*max_jitter 
    are returned centered at peak summit. This allows for jittering examples by randomly
    cropping. Data of width inputlen/outputlen is returned for validation
    data.

    If outliers is not None, removes training examples with counts > outlier%ile
    """

    cts_bw = pyBigWig.open(cts_bw_file)
    genome = pyfaidx.Fasta(genome_fasta)

    peak_regions_bed = process_bed(bed_regions).drop_duplicates()
    non_peak_regions_bed = process_bed(nonpeak_regions).drop_duplicates()

    train_peaks_seqs=None
    train_peaks_cts=None
    train_peaks_coords=None
    train_nonpeaks_seqs=None
    train_nonpeaks_cts=None
    train_nonpeaks_coords=None

    if bed_regions is not None:
        train_peaks_seqs, train_peaks_cts, train_peaks_coords = get_seq_cts_coords(peak_regions_bed,
                                              genome,
                                              cts_bw,
                                              inputlen+2*max_jitter,
                                              outputlen+2*max_jitter,
                                              peaks_bool=1)
    
    if nonpeak_regions is not None:
        train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords = get_seq_cts_coords(non_peak_regions_bed,
                                              genome,
                                              cts_bw,
                                              inputlen,
                                              outputlen,
                                              peaks_bool=0)



    cts_bw.close()
    genome.close()

    return (train_peaks_seqs, train_peaks_cts, train_peaks_coords,
            train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords)
