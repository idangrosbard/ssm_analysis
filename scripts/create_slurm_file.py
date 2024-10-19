import subprocess
from pathlib import Path
from typing import NamedTuple


def slurm_template(
        experiment_name: str,
        slurm_output_dir: Path,
        script_path: Path,
        args: dict,

) -> str:
    command_str = ' '.join([
        'python',
        f"{script_path.name}",
        *[
            f'--{k} "{v}"'
            for k, v in
            args.items()
        ]
    ])

    return (
        f"""\
#! /bin/bash

#SBATCH --job-name={experiment_name} # Job name
#SBATCH --output={slurm_output_dir / 'out.log'} # redirect stdout
#SBATCH --error={slurm_output_dir / 'err.log'} # redirect stderr
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
cd {str(script_path.parent)}

# Export environment variables
export PYTHONUNBUFFERED=1

{command_str}

# Trap keyboard interrupt (Ctrl+C) to end the job
trap 'echo "Keyboard interrupt received. Ending job."; exit' INT

""")


def create_template(
        experiment_name: str,
        output_dir: Path,
        script_path: Path,
        script_args: dict,
):
    """

    Returns:
        Path: Path to the slurm file

    """

    slurm_output_dir = output_dir / experiment_name

    slurm_output_dir.mkdir(parents=True, exist_ok=True)
    template = slurm_template(
        experiment_name=experiment_name,
        slurm_output_dir=slurm_output_dir,
        script_path=script_path,
        args=script_args
    )

    slurm_path = slurm_output_dir / f"run.slurm"

    slurm_path.write_text(template)

    return slurm_path


def run_slurm(
        experiment_name: str,
        output_dir: Path,
        script_path: Path,
        script_args: dict,
):
    slurm_path = create_template(
        experiment_name=experiment_name,
        output_dir=output_dir,
        script_path=script_path,
        script_args=script_args
    )
    print(f"Submitting job, to see the logs run: \nless +F {slurm_path.parent / 'err.log'}")
    subprocess.run(['sbatch', str(slurm_path)])
