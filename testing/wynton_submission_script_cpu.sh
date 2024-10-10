#$ -S /bin/bash
#$ -cwd               # job should run in the current working directory
#$ -j y               # STDERR and STDOUT should be joined
#$ -l mem_free=100MB     # job requires up to 1 GiB of RAM per slot
##$ -l scratch=300G      # job requires up to 2 GiB of local /scratch space
#$ -l h_rt=48:00:00   # job requires up to 24 hours of runtime
##$ -t 1-20           # array job with 10 tasks (remove first '#)


conda activate chrombpnet
trap 'conda deactivate' EXIT
## export CUDA_VISIBLE_DEVICES=$SGE_GPU

export WANDB_MODE=offline
python training_script.py --use_cpu=True


[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"

