#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -P neuroppg
#$ -l mem_free=20G,gpu_mem=20G
#$ -l h_rt=48:00:00
#$ -o /wynton/home/corces/vishvak/pytorch_cbp/testing/microglia_gpu_train/train_run_attention_005.out
#$ -e /wynton/home/corces/vishvak/pytorch_cbp/testing/microglia_gpu_train/train_run_attention_005.err
module load CBI miniconda3
module load samtools
conda activate chrombpnet
trap 'conda deactivate' EXIT
export CUDA_VISIBLE_DEVICES=$SGE_GPU
python training_script_microglia.py  --batch_size 64  --dilation_kernel_size 2 --out_pred_len 750  --n_dil_layers 11 --conv1_kernel_size 15 --filters 256 --input_seq_len 6000 --dropout_rate 0.1 --use_cpu False --use_attention_pooling True
