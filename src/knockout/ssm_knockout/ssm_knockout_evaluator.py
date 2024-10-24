from ..knockout_evaluator import KnockoutEvaluator
from ..knockout_mode import KnockoutMode
from typing import Iterable, Tuple
import pandas as pd
from .ssm_knockout_hook import SSMKnockoutHook
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from ...evaluate import evaluate_model


class SSMKnockoutEvaluator(KnockoutEvaluator):
    def __init__(self, model: MambaForCausalLM, tokenizer: AutoTokenizer, device: torch.device, knockout_ssm_indices: Iterable[Iterable[int]], show_progress: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.show_progress = show_progress
        self.knockout_ssm_indices = knockout_ssm_indices

    def setup_hooks(self, layers: Iterable[int], knockout_mode: KnockoutMode) -> Tuple[list, list]:
        hooks = []
        handles = []
        for i in range(len(self.model.backbone.layers)):
            if i in layers:
                # "mixer of interest" - moi
                moi = self.model.backbone.layers[i].mixer
                curr_knockout_indices = self.knockout_ssm_indices[i]
                d = moi.A_log.shape[0]
                hooks.append(SSMKnockoutHook(i, curr_knockout_indices, self.device, d))
                
                handles.append(moi.register_forward_hook(hooks[-1]))

        return hooks, handles

    def knockout_eval(self, dataset: pd.DataFrame, layers: Iterable[int], knockout_mode: KnockoutMode) -> Tuple[pd.DataFrame, int]:
        acc = 0
        hooks, handles = self.setup_hooks(layers, knockout_mode)

        # Evaluate model
        dataset, acc = evaluate_model(self.model, self.tokenizer, dataset, self.device)
        
        # remove hooks
        for handle in handles:
            handle.remove()

        return dataset, acc