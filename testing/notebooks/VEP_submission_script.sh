#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -P neuroppg
#$ -l mem_free=10G,gpu_mem=15G
#$ -l h_rt=2:00:00
#module load CBI miniconda3
module load cuda
conda activate chrombpnet
trap 'conda deactivate' EXIT
export CUDA_VISIBLE_DEVICES=$SGE_GPU
python VEP_script.py