import pyrallis

from src.consts import PATHS
from src.experiments.evaluate_model import EvaluateModelConfig, main_local
from src.types import DATASETS, MODEL_ARCH, DatasetArgs
from src.utils.slurm import submit_job


@pyrallis.wrap()
def main(args: EvaluateModelConfig):
    assert not (args.drop_subject and args.drop_subj_last_token)

    if args.with_slurm:
        gpu_type = "l40s"
        # gpu_type = "titan_xp-studentrun"

        for model_arch, model_size in [
            # (MODEL_ARCH.MAMBA1, "130M"),
            # (MODEL_ARCH.MAMBA1, "1.4B"),
            # (MODEL_ARCH.MAMBA1, "2.8B"),
            # (MODEL_ARCH.MAMBA2, "130M"),
            # (MODEL_ARCH.MAMBA2, "1.3B"),
            # (MODEL_ARCH.MAMBA2, "2.7B"),
            # (MODEL_ARCH.MINIMAL_MAMBA2, "130M"),
            # (MODEL_ARCH.MINIMAL_MAMBA2, "2.7B"),
            (MODEL_ARCH.MAMBA1, "7B"),
            # (MODEL_ARCH.MAMBA1, "7B-falcon"),
            # (MODEL_ARCH.MAMBA1, "7B-falcon-base"),
            # (MODEL_ARCH.LLAMA2, "7B"),
            # (MODEL_ARCH.LLAMA3_2, "1B"),
            # (MODEL_ARCH.LLAMA3_2, "3B"),
        ]:
            args.model_arch = model_arch
            args.model_size = model_size

            # for i in range(5):
            #     args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits=f"train{i+1}")
            # args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits=f"train1")
            args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all")
            # args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits=f"test")

            job_name = f"evaluate_model/{model_arch}_{model_size}_{args.dataset_args.dataset_name}"
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
