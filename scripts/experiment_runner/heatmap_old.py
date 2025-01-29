import pyrallis

from src.consts import PATHS
from src.experiments.heatmap import HeatmapConfig
from src.experiments.heatmap_old import main_local
from src.types import DATASETS, MODEL_ARCH, DatasetArgs
from src.utils.slurm import submit_job


@pyrallis.wrap()
def main(args: HeatmapConfig):
    if args.with_slurm:
        gpu_type = "l40s"
        # gpu_type = "titan_xp-studentrun"
        window_sizes = [1, 3, 5, 9, 12, 15]
        experiment_name = args.experiment_name
        # experiment_name = "heatmap_debug_use_matrix"
        variation_name = "_v8"
        args.experiment_name = experiment_name + variation_name
        # window_sizes = [1, 5]
        # window_sizes = [1, 5, 9]

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
                    f"{experiment_name}/{model_arch}_{model_size}_ws={window_size}_{args.dataset_args.dataset_name}"
                )

                job = submit_job(
                    main_local,
                    args,
                    log_folder=str(PATHS.SLURM_DIR / job_name / "%j"),
                    job_name=job_name,
                    # timeout_min=1200,
                    gpu_type=(
                        "l40s" if (model_size == "2.7B" and model_arch == MODEL_ARCH.MINIMAL_MAMBA2_new) else gpu_type
                    ),
                    slurm_gpus_per_node=(
                        1 if (model_size in ["2.8B", "2.7B"] and gpu_type == "titan_xp-studentrun") else 1
                    ),
                )

                print(f"{job}: {job_name}")
    else:
        args.experiment_name = "debug"
        args.prompt_indices = [1, 2, 3, 4, 5]
        main_local(args)


if __name__ == "__main__":
    main()
