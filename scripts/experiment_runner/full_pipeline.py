import pyrallis

from src.experiments.full_pipeline import FullPipelineConfig
from src.types import MODEL_ARCH


@pyrallis.wrap()
def main(args: FullPipelineConfig):
    # args.with_slurm = True
    # gpu_type = "titan_xp-studentrun"
    window_sizes = [9]
    # window_sizes = [1, 3, 5, 9, 12, 15]
    # experiment_name = "heatmap_debug_use_matrix"
    # args.variation = "v1_titan_xp"
    # window_sizes = [1, 5]
    # window_sizes = [1, 5, 9]

    for model_arch, model_size in [
        # (MODEL_ARCH.MAMBA1, "130M"),
        # (MODEL_ARCH.MAMBA1, "1.4B"),
        # (MODEL_ARCH.MAMBA1, "2.8B"),
        # (MODEL_ARCH.MAMBA1, "7B"),
        (MODEL_ARCH.MAMBA1, "7B-falcon"),
        (MODEL_ARCH.MAMBA1, "7B-falcon-base"),
        # (MODEL_ARCH.MAMBA2, "130M"),
        # (MODEL_ARCH.MAMBA2, "1.3B"),
        # (MODEL_ARCH.MAMBA2, "2.7B"),
        # (MODEL_ARCH.GPT2, "124M"),
        # (MODEL_ARCH.GPT2, "355M"),
        # (MODEL_ARCH.GPT2, "774M"),
        # (MODEL_ARCH.GPT2, "1.5B"),
    ]:
        args.model_arch = model_arch
        args.model_size = model_size
        # args.variation = "v3"
        # if model_arch == MODEL_ARCH.GPT2:
        #     args.variation = "v2"
        # args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all")
        for window_size in window_sizes:
            args.window_size = window_size

            if args.with_generation:
                args.slurm_gpus_per_node = (
                    1 if (model_size in ["2.8B", "2.7B"] and args.slurm_gpu_type == "titan_xp-studentrun") else 1
                )
                # args.slurm_gpu_type = (
                #     "l40s" if (model_size == "2.7B" and model_arch == MODEL_ARCH.MAMBA2) else args.slurm_gpu_type
                # )
            args.run()


if __name__ == "__main__":
    main()  # type: ignore
