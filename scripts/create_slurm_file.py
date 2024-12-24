import subprocess
from pathlib import Path

from src.consts import ISlurmArgs


def slurm_template(
    experiment_name: str,
    slurm_output_dir: Path,
    script_path: Path,
    args: dict,
    add_args: ISlurmArgs = ISlurmArgs,
) -> str:
    command_str = " ".join(["python", f"{script_path}", *[f'--{k} "{v}"' for k, v in args.items()]])

    return f"""\
#! /bin/bash

#SBATCH --job-name={experiment_name} # Job name
#SBATCH --output={slurm_output_dir / 'out.log'} # redirect stdout
#SBATCH --error={slurm_output_dir / 'err.log'} # redirect stderr
#SBATCH --partition={add_args.partition} # (see resources section)
#SBATCH --time={add_args.time} # max time (minutes)
#SBATCH --signal={add_args.singal} # how to end job when time’s up
#SBATCH --nodes={add_args.nodes} # number of machines
#SBATCH --ntasks={add_args.ntasks} # number of processes
#SBATCH --mem={add_args.mem} # CPU memory (MB)
#SBATCH --cpus-per-task={add_args.cpus_per_task} # CPU cores per process
#SBATCH --gpus={add_args.gpus} # GPUs in total
#SBATCH --account={add_args.account} # billing account

nvidia-smi
gpustat --no-color -pfu

# Activate the conda environment
source {add_args.workspace}/venv/bin/activate

# Print diagnostic information
echo $CUDA_VISIBLE_DEVICES
echo "Python version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo $(pwd)

# Change to the working directory
cd {add_args.workspace}

# Export environment variables
export PYTHONUNBUFFERED=1

{command_str}

# Trap keyboard interrupt (Ctrl+C) to end the job
trap 'echo "Keyboard interrupt received. Ending job."; exit' INT

"""


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
        args=script_args,
    )

    slurm_path = slurm_output_dir / "run.slurm"

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
        script_args=script_args,
    )
    print(f"Submitting job, to see the logs run: \nless +F {slurm_path.parent / 'err.log'}")
    subprocess.run(["sbatch", str(slurm_path)])
