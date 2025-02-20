from typing import Callable

import pyrallis

from src.consts import PATHS
from src.experiments.info_flow import InfoFlowConfig
from src.experiments.info_flow import main_local as main_local_info_flow
from src.types import DATASETS, MODEL_ARCH, DatasetArgs, TokenType
from src.utils.slurm import submit_job


@pyrallis.wrap()
def main(args: InfoFlowConfig, main_local: Callable[[InfoFlowConfig], None]):
    if args.with_slurm:
        gpu_type = "l40s"
        # gpu_type = "titan_xp-studentrun"

        args.experiment_name += "_v7"
        # window_sizes = [1, 3, 5, 9, 12, 15]
        window_sizes = [1, 3, 5, 9, 12, 15]
        # window_sizes = [9]
        # window_sizes = [12]
        # window_sizes = [12]

        # window_sizes =[1,2,3,4,5,6,7,8,9]

        # args.experiment_name += f"_test_top_outputs_5_last_windows"
        # args.knockout_map = {"last": ["subject"]}
        # args.knockout_map = {"last": ["last", "subject", "relation"]}
        # args.DEBUG_LAST_WINDOWS = 5
        # window_sizes = [9]

        for model_arch, model_size in [
            # (MODEL_ARCH.MAMBA1, "130M"),
            # (MODEL_ARCH.MAMBA1, "1.4B"),
            # (MODEL_ARCH.MAMBA1, "2.8B"),
            (MODEL_ARCH.MAMBA1, "7B"),
            # (MODEL_ARCH.MAMBA1, "7B-falcon"),
            (MODEL_ARCH.MAMBA1, "7B-falcon-base"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "130M"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "1.3B"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "2.7B"),
        ]:
            args.model_arch = model_arch
            args.model_size = model_size
            args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all")
            for window_size in window_sizes:
                args.window_size = window_size

                job_name = (
                    f"{args.experiment_name}/"
                    f"{model_arch}_{model_size}_ws={window_size}_{args.dataset_args.dataset_name}"
                )
                job = submit_job(
                    main_local,
                    args,
                    log_folder=str(PATHS.SLURM_DIR / job_name / "%j"),
                    job_name=job_name,
                    # timeout_min=1200,
                    gpu_type=gpu_type,
                    slurm_gpus_per_node=1,
                )

                print(f"{job}: {job_name}")
    else:
        args.experiment_name += "_debug"
        args.overwrite = True
        args.knockout_map = {TokenType.last: [TokenType.last, TokenType.subject, TokenType.relation]}
        args.DEBUG_LAST_WINDOWS = 1
        window_sizes = [9]
        main_local(args)


if __name__ == "__main__":
    main(main_local_info_flow)
    # main(main_local_info_flow_plot)
