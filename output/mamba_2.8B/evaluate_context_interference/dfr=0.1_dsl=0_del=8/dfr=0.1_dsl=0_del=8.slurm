#! /bin/bash

#SBATCH --job-name=M_2.8B_eci_ID_dfr=0.1_dsl=0_del=8 # Job name
#SBATCH --output=/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/mamba_2.8B/evaluate_context_interference/dfr=0.1_dsl=0_del=8/out.log # redirect stdout
#SBATCH --error=/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/mamba_2.8B/evaluate_context_interference/dfr=0.1_dsl=0_del=8/err.log # redirect stderr
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

python evaluate_context_interference.py --interfere_mode INCREASE_DELTA --output_dir '/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/mamba_2.8B/evaluate_context_interference/dfr=0.1_dsl=0_del=8' --model_size 2.8B  --delta_factor_root 0.1 --delta_start_layer 0 --delta_end_layer 8


# Trap keyboard interrupt (Ctrl+C) to end the job
trap 'echo "Keyboard interrupt received. Ending job."; exit' INT



