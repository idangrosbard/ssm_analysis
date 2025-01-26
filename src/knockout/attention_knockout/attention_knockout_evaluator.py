from typing import Iterable, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, MambaForCausalLM

from src.knockout.knockout_evaluator import KnockoutEvaluator
from src.knockout.knockout_mode import KnockoutMode

from .knockout_target import KnockoutTarget
from .knockout_target_calc import choose_knockout_target, is_last_token_subj
from .ssm_interfere import SSMInterfereHook


class AttentionKnockoutEvaluator(KnockoutEvaluator):
    def __init__(
        self,
        model: MambaForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        knockout_target: KnockoutTarget,
        affected_target: KnockoutTarget,
        drop_subj_last: bool = False,
        show_progress: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.knockout_target = knockout_target
        self.affected_target = affected_target
        self.drop_subj_last = drop_subj_last
        self.show_progress = show_progress

    def knockout_eval(
        self, dataset: pd.DataFrame, layers: Iterable[int], knockout_mode: KnockoutMode
    ) -> Tuple[pd.DataFrame, int]:
        acc = 0
        hooks = []
        handles = []
        dataset["correct"] = False

        # set up hooks
        for i in range(len(self.model.backbone.layers)):
            if i in layers:
                # "mixer of interest" - moi
                moi = self.model.backbone.layers[i].mixer

                hooks.append(SSMInterfereHook(i, knockout_mode))

                handles.append(moi.register_forward_hook(hooks[-1]))

        # Evaluate model
        pbar = tqdm(dataset.index, total=len(dataset), disable=not self.show_progress)
        for idx in pbar:
            # Get relevant data
            input = dataset.loc[idx, "prompt"]
            target = dataset.loc[idx, "attribute"]
            subj = dataset.loc[idx, "subject"]

            if self.drop_subj_last:
                if is_last_token_subj(input, subj, self.tokenizer):
                    continue

            # set subject token as knockout idx
            knockout_indices = choose_knockout_target(input, subj, self.tokenizer, self.knockout_target)
            affected_target_indices = choose_knockout_target(input, subj, self.tokenizer, self.affected_target)

            for hook in hooks:
                hook.knockout_indices = knockout_indices
                hook.affected_outputs = affected_target_indices

            input_ids = self.tokenizer(input, return_tensors="pt")["input_ids"].to(self.device)
            out = self.model(input_ids)

            # get last decoded word
            decoded = self.tokenizer.decode(out.logits.argmax(dim=-1).squeeze())
            last_word = decoded.split(" ")[-1]

            correct = last_word == target[: len(last_word)]
            # Update performance
            acc += float(correct) / len(dataset)
            dataset.loc[idx, "correct"] = correct

        # remove hooks
        for handle in handles:
            handle.remove()

        return dataset, acc
