#! /bin/bash

#SBATCH --job-name=mamba_2.8B_evaluate_context_interference_ZERO_ATTENTION # Job name
#SBATCH --output=/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/mamba_2.8B/evaluate_context_interference/ZERO_ATTENTION/out.log # redirect stdout
#SBATCH --error=/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/mamba_2.8B/evaluate_context_interference/ZERO_ATTENTION/err.log # redirect stderr
#SBATCH --partition=gpu-a100-killable # (see resources section)
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=8G # CPU memory (MB)
#SBATCH --cpus-per-task=1 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
#SBATCH --account=gpu-research # billing account

nvidia-smi
gpustat --no-color -pfu

# Activate the conda environment
source /home/yandex/DL20232024a/nirendy/repos/ADL_2/venv/bin/activate

# Print diagnostic information
echo $CUDA_VISIBLE_DEVICES
echo "Python version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo $(pwd)

# Change to the working directory
cd /home/yandex/DL20232024a/nirendy/repos/ssm_analysis/scripts

# Export environment variables
export PYTHONUNBUFFERED=1

python evaluate_context_interference.py --interfere_mode ZERO_ATTENTION --output_dir '/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/mamba_2.8B/evaluate_context_interference/ZERO_ATTENTION' --model_size 2.8B --affected_output last

# Trap keyboard interrupt (Ctrl+C) to end the job
trap 'echo "Keyboard interrupt received. Ending job."; exit' INT

