from enum import StrEnum

import submitit

"""
| GPU                       | Speed (TFLOPS) | Memory (GB) |
|---------------------------|----------------|-------------|
| NVIDIA H100-80GB HBM3     | 65.0           | 80          |
| NVIDIA A100-SXM-80GB      | 19.5           | 80          |
| L40S                      | 32.0           | 48          |
| A6000                     | 22.3           | 48          |
| Quadro RTX 8000           | 16.3           | 48          |
| NVIDIA GeForce RTX 3090   | 35.6           | 24          |
| A5000                     | 24.0           | 24          |
| Tesla V100-SXM2-32GB      | 15.7           | 32          |
| NVIDIA GeForce RTX 2080 Ti| 13.4           | 11          |
| Nvidia Titan XP           | 12.1           | 12          |
"""


class SLURM_GPU_TYPE(StrEnum):
    TITAN_XP_STUDENTRUN = "titan_xp-studentrun"
    L40S = "l40s"
    A100 = "a100"
    H100 = "h100"
    GEFORCE_RTX_3090 = "geforce_rtx_3090"
    V100 = "v100"
    A5000 = "a5000"
    A6000 = "a6000"
    QUADRO_RTX_8000 = "quadro_rtx_8000"
    TESLA_V100_SXM2_32GB = "tesla_v100_sxm2_32gb"
    TITAN_XP_STUDENTRUN_BATCH = "titan_xp-studentrun-batch"
    TITAN_XP_STUDENTRUN_KILLABLE = "titan_xp-studentrun-killable"

    @property
    def gpu_name(self) -> str:
        match self:
            case (
                SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN
                | SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN_BATCH
                | SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN_KILLABLE
            ):
                return "titan_xp"
            case _:
                return self.value


def submit_job(
    func,
    *args,
    gpu_type: SLURM_GPU_TYPE,
    job_name="test",
    log_folder="log_test/%j",  # %j is replaced by the job id at runtime
    timeout_min=1200,
    memory_required=None,
    slurm_nodes=1,
    tasks_per_node=1,
    slurm_cpus_per_task=1,
    slurm_gpus_per_node=0,
    slurm_nodelist=None,
):
    # Map GPU type and account type to partition and account options based on `sinfo` data
    partition_account_map = {
        SLURM_GPU_TYPE.GEFORCE_RTX_3090: {"partition": "killable", "account": "gpu-students"},
        SLURM_GPU_TYPE.V100: {"partition": "killable", "account": "gpu-students"},
        SLURM_GPU_TYPE.A5000: {"partition": "killable", "account": "gpu-students"},
        SLURM_GPU_TYPE.A6000: {"partition": "killable", "account": "gpu-research"},
        SLURM_GPU_TYPE.L40S: {"partition": "killable", "account": "gpu-research"},
        SLURM_GPU_TYPE.A100: {"partition": "gpu-a100-killable", "account": "gpu-research"},
        SLURM_GPU_TYPE.H100: {"partition": "gpu-h100-killable", "account": "gpu-research"},
        SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN: {
            "partition": "studentrun",
            "account": "gpu-students",
            "nodelist": "s-003, s-004, s-005",
        },
        SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN_BATCH: {
            "partition": "studentbatch",
            "account": "gpu-students",
            "nodelist": "s-003, s-004, s-005",
        },
        SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN_KILLABLE: {
            "partition": "studentkillable",
            "account": "gpu-students",
            "nodelist": "s-003, s-004, s-005",
        },
    }

    if gpu_type == SLURM_GPU_TYPE.TITAN_XP_STUDENTRUN:
        timeout_min = 150

    # Determine the appropriate partition and account based on `gpu_type`
    partition_account = partition_account_map[gpu_type]
    slurm_partition = partition_account["partition"]
    slurm_account = partition_account["account"]
    slurm_nodelist = slurm_nodelist or partition_account.get("nodelist", slurm_nodelist)

    def ommit_none(d):
        return {k: v for k, v in d.items() if v is not None}

    # Setup the executor
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name=job_name,
        timeout_min=timeout_min,
        slurm_partition=slurm_partition,
        slurm_account=slurm_account,
        slurm_nodes=slurm_nodes,
        tasks_per_node=tasks_per_node,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_gpus_per_node=slurm_gpus_per_node,
        slurm_mem=memory_required,
        # slurm_constraint=gpu_type.gpu_name,
        **ommit_none(
            dict(
                slurm_nodelist=slurm_nodelist,
            )
        ),
    )

    # Submit the job
    job = executor.submit(func, *args)
    return job


def submit_cpu_job(
    func,
    *args,
    job_name="test",
    log_folder="log_test/%j",  # %j is replaced by the job id at runtime
    timeout_min=1200,
    memory_required=None,
    slurm_nodes=1,
    tasks_per_node=1,
    slurm_cpus_per_task=1,
):
    # Map GPU type and account type to partition and account options based on `sinfo` data
    partition_account = "studentbatch"
    account = "gpu-students"

    # Setup the executor
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name=job_name,
        timeout_min=timeout_min,
        slurm_partition=partition_account,
        slurm_account=account,
        slurm_nodes=slurm_nodes,
        tasks_per_node=tasks_per_node,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=memory_required,
    )

    # Submit the job
    job = executor.submit(func, *args)
    return job
