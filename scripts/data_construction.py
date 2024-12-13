from collections import defaultdict
from dataclasses import dataclass
from hmac import new
import json
from pathlib import Path
from typing import Optional, assert_never
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
import pyrallis
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)

from scripts.create_slurm_file import run_slurm
from src.consts import (
    FILTERATIONS,
    MODEL_SIZES_PER_ARCH_TO_MODEL_ID,
    PATHS,
)
from src.datasets.download_dataset import load_dataset, load_splitted_counter_fact
from src.datasets.download_dataset import load_knowns_pd
from src.logit_utils import get_last_token_logits, logits_to_probs
from src.models.model_interface import get_model_interface
from src.types import DATASETS
from src.types import MODEL_ARCH, SPLIT, DatasetArgs, TModelID
from src.utils.setup_models import get_tokenizer_and_model
from src.utils.slurm import submit_job


@dataclass
class Args:
    # model_arch: MODEL_ARCH = MODEL_ARCH.MINIMAL_MAMBA2_new
    model_arch: MODEL_ARCH = MODEL_ARCH.MAMBA1
    model_size: str = "130M"
    dataset_args: DatasetArgs = pyrallis.field(
        default=DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all"), is_mutable=True
    )
    filteration: str = FILTERATIONS.all_correct
    _batch_size: int = 16  # Adjust based on GPU memory
    output_file: Optional[Path] = None
    with_slurm: bool = False
    temperature = 1
    top_k = 0
    top_p = 1
    output_dir: Optional[Path] = None
    attention: bool = False

    @property
    def batch_size(self) -> int:
        return (
            1
            if (
                self.model_arch == MODEL_ARCH.MINIMAL_MAMBA2
                or self.model_arch == MODEL_ARCH.MINIMAL_MAMBA2_new
            )
            else self._batch_size
        )

    @property
    def model_id(self) -> TModelID:
        return MODEL_SIZES_PER_ARCH_TO_MODEL_ID[self.model_arch][self.model_size]


def main_local(args: Args):
    print(args)
    original_data = pd.DataFrame(
        load_splitted_counter_fact(
            "all", align_to_known=False, filteration=args.filteration
        )
    )

    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    attention = args.attention

    if not args.output_file:
        args.output_file = (
            PATHS.OUTPUT_DIR
            / args.model_id
            / "data_construction"
            / f"ds={args.dataset_args.dataset_name}"
        )
        
    args.output_file.mkdir(parents=True, exist_ok=True)

    
    model_interface = get_model_interface(args.model_arch, args.model_size)
    tokenizer = model_interface.tokenizer
    device = model_interface.device
    
    original_data['true_prob'] = 0.0
    original_data['max_prob'] = 0.0
    original_data['hit'] = False
    original_data['pred'] = ""
    
    def forward_eval(temperature, top_k, top_p, batch_start, batch_end, attention, print_period=10000):
        prompts = list(original_data.loc[batch_start:batch_end-1, 'prompt'].values)
        true_word = list(original_data.loc[batch_start:batch_end-1, 'target_true'].values)
        true_token = tokenizer(true_word, return_tensors="pt", padding=True)
        true_id = true_token.input_ids.to(device='cpu')
        tokens = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = tokens.input_ids.to(device=device)
        max_new_length = input_ids.shape[1] + 1
        next_token_probs = model_interface.generate_logits(
            input_ids=input_ids,
            attention=attention,
        )
        max_idx = np.argmax(next_token_probs, axis=1)
        row_idx = np.arange(next_token_probs.shape[0])
        preds = [tokenizer.decode([t]) for t in max_idx]
        original_data.loc[batch_start:batch_end-1, 'true_prob'] = next_token_probs[row_idx, true_id[:, 0]]
        original_data.loc[batch_start:batch_end-1, 'max_prob'] = next_token_probs[row_idx, max_idx]
        original_data.loc[batch_start:batch_end-1, 'hit'] = original_data.loc[batch_start:batch_end-1, 'true_prob'] == original_data.loc[batch_start:batch_end-1, 'max_prob']
        original_data.loc[batch_start:batch_end-1, 'pred'] = preds
        if (batch_start+1) % print_period == 0:
            print(f'Finished batch [{batch_start}:{batch_end-1}]')
        torch.cuda.empty_cache()
        
    batch_size = 1
    N = len(original_data)
    batches = list(np.arange(0, N, batch_size)) + [N]
    #%%
    for i in tqdm(range(len(batches)-1)):
        forward_eval(temperature, top_k, top_p, batches[i], batches[i+1], attention)

    print(original_data.pipe(lambda df: df[~df['hit']]))
    print(original_data['hit'].mean())
    attention_str = "attention" if attention else "original"
    original_data.to_parquet(args.output_file / f'entire_results_{attention_str}.parquet')


@pyrallis.wrap()
def main(args: Args):
    # args.with_slurm = True

    if args.with_slurm:
        gpu_type = "a100"
        # gpu_type = "titan_xp-studentrun"

        for model_arch, model_size in [
            # (MODEL_ARCH.MAMBA1, "130M"),
            # (MODEL_ARCH.MAMBA1, "1.4B"),
            # (MODEL_ARCH.MAMBA1, "2.8B"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "130M"),
            # (MODEL_ARCH.MINIMAL_MAMBA2_new, "1.3B"),
            (MODEL_ARCH.MINIMAL_MAMBA2_new, "2.7B"),
        ]:
            args.model_arch = model_arch
            args.model_size = model_size
            args.dataset_args = DatasetArgs(name=DATASETS.COUNTER_FACT, splits=f"all")
            for attention in [True, False]:
                args.attention = attention

                job_name = f"data_construction/{model_arch}_{model_size}_attention={attention}_{args.dataset_args.dataset_name}"
                job = submit_job(
                    main_local,
                    args,
                    log_folder=str(PATHS.SLURM_DIR / job_name / "%j"),
                    job_name=job_name,
                    # timeout_min=1200,
                    gpu_type=gpu_type,
                    slurm_gpus_per_node=3,
                )

                print(f"{job}: {job_name}")
    else:
        main_local(args)


if __name__ == "__main__":
    main()
