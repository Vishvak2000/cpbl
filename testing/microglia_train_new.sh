#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -P neuroppg
#$ -l mem_free=20G,gpu_mem=20G
#$ -l h_rt=48:00:00
#module load CBI miniconda3
module load cuda
conda activate chrombpnet
trap 'conda deactivate' EXIT
export CUDA_VISIBLE_DEVICES=$SGE_GPU
python training_script_microglia_new.py --filters 512  --dilation_kernel_size 3 --out_pred_len 1000  --n_dil_layers 8 --conv1_kernel_size 21 --input_seq_len 2114 --dropout_rate 0.0 --use_cpu False "$@"
