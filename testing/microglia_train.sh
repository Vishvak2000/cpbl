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
python training_script_microglia.py --loss mse --batch_size 64  --dilation_kernel_size 2 --out_pred_len 750  --n_dil_layers 11 --conv1_kernel_size 8 --filters 64 --input_seq_len 2000 --dropout_rate 0.05 --use_cpu False "$@"
