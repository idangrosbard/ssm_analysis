#! /bin/bash

#SBATCH --job-name=minimal_mamba2_130M # Job name
#SBATCH --output=/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/minimal_mamba2_130M/evaluate/minimal_mamba2_130M/out.log # redirect stdout
#SBATCH --error=/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/minimal_mamba2_130M/evaluate/minimal_mamba2_130M/err.log # redirect stderr
#SBATCH --partition=gpu-a100-killable # (see resources section)
#SBATCH --time=1200 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
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

python evaluate_model.py --model_arch "minimal_mamba2" --model_size "130M" --with_3_dots "0" --output_file "/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/minimal_mamba2_130M/evaluate/known_1000_correct.csv"

# Trap keyboard interrupt (Ctrl+C) to end the job
trap 'echo "Keyboard interrupt received. Ending job."; exit' INT

