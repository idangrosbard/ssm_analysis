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


def submit_job(
    func,
    *args,
    gpu_type,
    job_name="test",
    log_folder="log_test/%j",  # %j is replaced by the job id at runtime
    timeout_min=1200,
    memory_required=None,
    slurm_nodes=1,
    tasks_per_node=1,
    slurm_cpus_per_task=1,
    slurm_gpus_per_node=1,
    slurm_nodelist=None,
):
    # Map GPU type and account type to partition and account options based on `sinfo` data
    partition_account_map = {
        "geforce_rtx_3090": {"partition": "killable", "account": "gpu-students"},
        "v100": {"partition": "killable", "account": "gpu-students"},
        "a5000": {"partition": "killable", "account": "gpu-students"},
        "a6000": {"partition": "killable", "account": "gpu-research"},
        "l40s": {"partition": "killable", "account": "gpu-research"},
        "a100": {"partition": "gpu-a100-killable", "account": "gpu-research"},
        "h100": {"partition": "gpu-h100-killable", "account": "gpu-research"},
        "titan_xp-studentrun": {
            "partition": "studentrun",
            "account": "gpu-students",
            "nodelist": "s-003, s-004, s-005",
        },
        "titan_xp-studentbatch": {
            "partition": "studentbatch",
            "account": "gpu-students",
            "nodelist": "s-003, s-004, s-005",
        },
        "titan_xp-studentkillable": {
            "partition": "studentkillable",
            "account": "gpu-students",
            "nodelist": "s-003, s-004, s-005",
        },
    }

    if gpu_type == "titan_xp-studentrun":
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
        slurm_constraint=gpu_type.split("-")[0] if gpu_type else None,
        **ommit_none(
            dict(
                slurm_nodelist=slurm_nodelist,
            )
        ),
    )

    # Submit the job
    job = executor.submit(func, *args)
    return job
