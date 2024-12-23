import argparse
import torch
import torch.nn as nn
import sys
import os
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger



# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Training script for CBPLTrainer.')
    
    # Add arguments with default values
    parser.add_argument('--peak_regions', type=str, default="/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/pd_data/Microglia_peak_set_2.bed",
                        help='Path to peak regions BED file')
    parser.add_argument('--nonpeak_regions', type=str, default="/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/own_data/test.chr1.negatives.adjusted.bed",
                        help='Path to non-peak regions BED file')
    parser.add_argument('--genome_fasta', type=str, default="/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/data/downloads/hg38.fa",
                        help='Path to genome FASTA file')
    parser.add_argument('--cts_bw_file', type=str, default="/gladstone/corces/lab/users/vishvak/chrombpnet_tutorial/pd_data/nd_Microglia_merge.bw",
                        help='Path to counts BigWig file')
    parser.add_argument('--negative_sampling_ratio', type=float, default=0,
                        help='Ratio for negative sampling')
    parser.add_argument('--train_size', type=float, default=0.6,
                        help='Training data size as a fraction of the total dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--filters', type=int, default=512,
                        help='Number of filters in the convolutional layers')
    parser.add_argument('--n_dil_layers', type=int, default=9,
                        help='Number of dilated convolutional layers')
    parser.add_argument('--conv1_kernel_size', type=int, default=21,
                        help='Kernel size for the first convolutional layer')
    parser.add_argument('--profile_kernel_size', type=int, default=71,
                        help='Kernel size for the profile convolutional layer')
    parser.add_argument('--dilation_kernel_size', type=int, default=3,
                        help='Kernel size for the dilated convolutional layers')
    parser.add_argument('--input_seq_len', type=int, default=3000,
                        help='Input sequence length')
    parser.add_argument('--out_pred_len', type=int, default=750,
                        help='Output prediction length')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--train_chrs', type=list, default=["chr2","chr4","chr5","chr7","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr21","chr22","chrX","chrY"],
                        help='List for training chrs')
    parser.add_argument('--valid_chrs', type=list, default=["chr1","chr3","chr5","chr6","chr8"],
                        help='List for validation chrs')
    parser.add_argument('--seq_focus_len', type=int, default=500,
                        help='Gaussian middle n weighting')
    parser.add_argument('--use_cpu',type=bool,default=False,
                        help='Using CPU or GPU')
    parser.add_argument('--multinomial_weight',type=float,default=1.0,
                        help='mnll loss weighting')
    parser.add_argument('--mse_weight',type=float,default=1.0,
                        help='mse loss weighting')
    parser.add_argument('--checkpoint_path',type=str,default=None,
                        help="model checkpoint path") 
    parser.add_argument('--project', type=str, default="cbpl_new_microglia",
                        help='Project name for wandb')  
    parser.add_argument('--alpha',type=float,default=1.0,
                        help="MSE weighting parameter") 
    parser.add_argument('--flavor',type=str,default=None,
                        help="Combined loss type (MSE = mse or NLL=None)")  
    parser.add_argument('--run_name',type=str,default=None,
                        help="Run Name for wandb")
    parser.add_argument('--return_embeddings',type=bool,default=False,
                        help="return embeddings boolean")    
    parser.add_argument('--jitter',type=bool,default=True,
                        help="include jitter or not")           
    parser.add_argument('--jitter_scale',type=float,default=0.1,
                        help="alpha parameter for jittering")       
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
   

    # Set the number of threads to use (30% of available CPU cores)
    if args.use_cpu:
        total_cores = os.cpu_count()
        num_cores_to_use = max(1, int(total_cores * 0.3))  # Ensure at least one core is used
        torch.set_num_threads(num_cores_to_use)

    # Add the parent directory to the Python path
    sys.path.append(os.path.abspath(os.path.join('..')))

    from models._model import CBPLTrainer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the config dictionary from parsed arguments
    config = {
        "peak_regions": args.peak_regions,
        "nonpeak_regions": args.nonpeak_regions,
        "genome_fasta": args.genome_fasta,
        "cts_bw_file": args.cts_bw_file,
        "negative_sampling_ratio": args.negative_sampling_ratio,
        "train_size": args.train_size,
        "batch_size": args.batch_size,
        "filters": args.filters,
        "n_dil_layers": args.n_dil_layers,
        "conv1_kernel_size": args.conv1_kernel_size,
        "profile_kernel_size" : args.profile_kernel_size,
        "dilation_kernel_size": args.dilation_kernel_size,
        "input_seq_len": args.input_seq_len,
        "out_pred_len": args.out_pred_len,
        "dropout_rate": args.dropout_rate,
        "learning_rate": args.learning_rate,
        "train_chrs" : args.train_chrs,
        "valid_chrs" : args.valid_chrs,
        "seq_focus_len" : args.seq_focus_len,
        #"multinomial_weight" : args.multinomial_weight,
        #"mse_weight" : args.mse_weight,
        "alpha" : args.alpha,
        "flavor" : args.flavor,
        "return_embeddings": args.return_embeddings,
        "jitter" : args.jitter,
        "jitter_scale": args.jitter_scale
    }

    wandb_logger = WandbLogger(project=args.project,config=config,mode='offline',name=args.run_name)

    if args.checkpoint_path is not None:
        print(f"Loading in checkpoint {args.checkpoint_path}")
        trainer = CBPLTrainer(config,checkpoint_path=args.checkpoint_path)
    else:
        trainer = CBPLTrainer(config)

    # Optional: Print model architecture for verification
    print(trainer.model)

    summary(trainer.model, input_size=(args.batch_size, 4, args.input_seq_len))

    trainer.fit(logger_out=wandb_logger)
    

if __name__ == "__main__":
    main()
