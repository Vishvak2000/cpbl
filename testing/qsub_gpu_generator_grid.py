import numpy as np
import subprocess
import time
import os
from datetime import datetime
from itertools import product

def generate_hyperparameters():
    param_grid = {
        'batch_size': [64, 128, 256, 512],
        'dilation_kernel_size': [2, 5, 9],
        'out_pred_len': [750],
        'n_dil_layers': [1, 6, 11],
        'conv1_kernel_size': [5, 10, 15],
        'filters': [16, 64, 256],
        'input_seq_len': [2000, 4000, 6000],
        'dropout_rate': [0.0, 0.1, 0.2]
    }
    return [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

def create_qsub_script(params, run_id, log_dir):
    cwd = os.getcwd()
    
    script = f"""#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l mem_free=30G
#$ -l h_rt=48:00:00
#$ -o {os.path.join(cwd, log_dir, f'train_{run_id}.out')}
#$ -e {os.path.join(cwd, log_dir, f'train_{run_id}.err')}
module load CBI miniconda3
module load samtools
conda activate chrombpnet
trap 'conda deactivate' EXIT
export CUDA_VISIBLE_DEVICES=$SGE_GPU
python training_script.py \\
    --batch_size {params['batch_size']} \\
    --dilation_kernel_size {params['dilation_kernel_size']} \\
    --out_pred_len {params['out_pred_len']} \\
    --n_dil_layers {params['n_dil_layers']} \\
    --conv1_kernel_size {params['conv1_kernel_size']} \\
    --filters {params['filters']} \\
    --input_seq_len {params['input_seq_len']} \\
    --dropout_rate {params['dropout_rate']:.6f} \\
    --project cpbl_gpu_sweep
"""
    return script

def main():
    all_params = generate_hyperparameters()
    num_trials = len(all_params)
    max_concurrent = 16  # Maximum number of concurrent jobs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'gpu_sweep_logs_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    script_dir = os.path.join(log_dir, 'job_scripts')
    os.makedirs(script_dir, exist_ok=True)
    
    active_jobs = set()
    for trial, params in enumerate(all_params):
        run_id = f'run_{trial:03d}'
        
        script_path = os.path.join(script_dir, f'job_{run_id}.sh')
        with open(script_path, 'w') as f:
            f.write(create_qsub_script(params, run_id, log_dir))
        os.chmod(script_path, 0o755)
        
        while len(active_jobs) >= max_concurrent:
            completed = set()
            for job_id in active_jobs:
                result = subprocess.run(['qstat', '-j', job_id], capture_output=True)
                if result.returncode != 0:
                    completed.add(job_id)
            active_jobs -= completed
            if len(active_jobs) >= max_concurrent:
                time.sleep(60)
        
        result = subprocess.run(['qsub', '-q', 'gpu.q', script_path], capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[2]
            active_jobs.add(job_id)
            print(f"Submitted job {run_id} (job ID: {job_id})")
        else:
            print(f"Error submitting job {run_id}: {result.stderr}")
    
    print(f"\nAll jobs submitted. Logs will be available in: {log_dir}")

if __name__ == '__main__':
    main()