from .. import KnockoutEvaluator
from .. import KnockoutMode
from typing import Iterable, Tuple
import pandas as pd
from tqdm import tqdm
from .hook import IncreaseDeltaHook
from ..attention_knockout.knockout_target import KnockoutTarget
from ..attention_knockout.knockout_target_calc import choose_knockout_target, is_last_token_subj
from transformers import AutoTokenizer, MambaForCausalLM
import torch


def indices2khot(indices: Iterable[int], len: int, flip: bool = True) -> torch.Tensor:
    if type(indices) is not torch.Tensor:
        if type(indices) is not list:
            indices = list(indices)
        indices = torch.tensor(indices, dtype=torch.long)
    one_hots = torch.nn.functional.one_hot(indices, len)
    k_hot = one_hots.sum(dim=0)
    if flip:
        k_hot = 1 - k_hot
    return k_hot

class IncreaseDeltaEvaluator(KnockoutEvaluator):
    def __init__(self, model: MambaForCausalLM, tokenizer: AutoTokenizer, device: torch.device, affected_tokens: KnockoutTarget, feature_map: Iterable[torch.Tensor] | Iterable[Iterable[int]], factor: float = 1.5, show_progress: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.show_progress = show_progress
        self.feature_map = feature_map
        self.factor = factor
        self.affected_tokens = affected_tokens

    def setup_hooks(self, layers) -> Tuple[Iterable[IncreaseDeltaHook], Iterable[torch.utils.hooks.RemovableHandle]]:
        hooks = []
        handles = []

        # set up hooks
        for i in range(len(self.model.backbone.layers)):
            if i in layers:
                # "mixer of interest" - moi
                moi = self.model.backbone.layers[i].mixer

                feature_map = indices2khot(self.feature_map[i], moi.A_log.shape[0])

                hooks.append(IncreaseDeltaHook(i, feature_map, self.factor))
                
                handles.append(moi.register_forward_hook(hooks[-1]))
        
        return hooks, handles


    def knockout_eval(self, dataset: pd.DataFrame, layers: Iterable[int], knockout_mode: KnockoutMode) -> Tuple[pd.DataFrame, int]:
        acc = 0
        
        dataset['correct'] = False
        hooks, handles = self.setup_hooks(layers)
                
        # Evaluate model
        pbar = tqdm(dataset.index, total=len(dataset), disable=not self.show_progress)
        for idx in pbar:
            # Get relevant data
            input = dataset.loc[idx, "prompt"]
            target = dataset.loc[idx, "attribute"]
            subj = dataset.loc[idx, "subject"]
            

            input_ids = self.tokenizer(input, return_tensors="pt")["input_ids"].to(self.device)
            

            # set subject token as knockout idx
            affected_tokens = list(choose_knockout_target(input, subj, self.tokenizer, self.affected_tokens))
            # drop max token
            affected_tokens = sorted(affected_tokens)

            affected_tokens = indices2khot(affected_tokens, input_ids.shape[-1])
            for hook in hooks:
                hook.factored_tokens = affected_tokens

            out = self.model(input_ids)

            # get last decoded word
            decoded = self.tokenizer.decode(out.logits.argmax(dim=-1).squeeze())
            last_word = decoded.split(' ')[-1]

            correct = last_word == target[:len(last_word)]
            # Update performance
            acc += float(correct) / len(dataset)
            dataset.loc[idx, 'correct'] = correct
        
        # remove hooks
        for handle in handles:
            handle.remove()

        return dataset, acc