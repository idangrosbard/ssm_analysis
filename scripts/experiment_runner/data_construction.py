import pyrallis

from src.consts import PATHS
from src.experiments.data_construction import DataConstructionConfig, main_local
from src.types import DATASETS, MODEL_ARCH, DatasetArgs
from src.utils.slurm import submit_job


@pyrallis.wrap()
def main(args: DataConstructionConfig):
    if args.with_slurm:
        gpu_type = "l40s"
        # gpu_type = "titan_xp-studentrun"

        for model_arch, model_size in [
            # (MODEL_ARCH.MAMBA1, "130M"),
            # (MODEL_ARCH.MAMBA1, "1.4B"),
            # (MODEL_ARCH.MAMBA1, "2.8B"),
            (MODEL_ARCH.MAMBA1, "7B"),
            (MODEL_ARCH.MAMBA1, "7B-falcon"),
            (MODEL_ARCH.MAMBA1, "7B-falcon-base"),
            # (MODEL_ARCH.MINIMAL_MAMBA2, "130M"),
            # (MODEL_ARCH.MINIMAL_MAMBA2, "1.3B"),
            # (MODEL_ARCH.MINIMAL_MAMBA2, "2.7B"),
        ]:
            args.model_arch = model_arch
            args.model_size = model_size
            args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all")
            for attention in [True, False]:
                args.attention = attention

                job_name = (
                    f"data_construction/{model_arch}_{model_size}"
                    f"_attention={attention}_{args.dataset_args.dataset_name}"
                )
                job = submit_job(
                    main_local,
                    args,
                    log_folder=str(PATHS.SLURM_DIR / job_name / "%j"),
                    job_name=job_name,
                    gpu_type=gpu_type,
                    slurm_gpus_per_node=1,
                )

                print(f"{job}: {job_name}")
    else:
        main_local(args)


if __name__ == "__main__":
    main()  # type: ignore
