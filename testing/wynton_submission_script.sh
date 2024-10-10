#$ -S /bin/bash
#$ -cwd               # job should run in the current working directory
#$ -j y               # STDERR and STDOUT should be joined
#$ -l mem_free=30G     # job requires up to 1 GiB of RAM per slot
##$ -l scratch=300G      # job requires up to 2 GiB of local /scratch space
#$ -l h_rt=48:00:00   # job requires up to 24 hours of runtime
##$ -t 1-20           # array job with 10 tasks (remove first '#)

module load CBI miniconda3'
module load samtools

conda activate chrombpnet
trap 'conda deactivate' EXIT
export CUDA_VISIBLE_DEVICES=$SGE_GPU


python training_script.py --use_cpu=False


[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"

