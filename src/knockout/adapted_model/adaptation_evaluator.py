from .. import KnockoutEvaluator
from .. import KnockoutMode
from typing import Iterable, Tuple, Dict
import pandas as pd
from tqdm import tqdm
from ..attention_knockout.knockout_target_calc import choose_knockout_target, is_last_token_subj
from transformers import AutoTokenizer, MambaForCausalLM, MambaModel
import torch
from .adapted_mixer import AdaptedMixer


def indices2khot(indices: Iterable[int], len: int, flip: bool = True) -> torch.Tensor:
    if type(indices) is not torch.Tensor:
        if type(indices) is not list:
            indices = list(indices)
        print(indices)
        indices = torch.tensor(indices, dtype=torch.long)
    one_hots = torch.nn.functional.one_hot(indices, len)
    k_hot = one_hots.sum(dim=0)
    if flip:
        k_hot = 1 - k_hot
    return k_hot


def build_delta_factor_map(layers: Dict[str, Iterable[int]], factor: Dict[str, float], size: int, layer_index: int) -> Dict[str, float]:
    factor_maps = []
    for layer, indices in layers.items():
        feature_map = indices2khot(indices[layer_index], size, flip=False)
        feature_map = feature_map.unsqueeze(0)
        factor_maps.append(feature_map * factor[layer])
    delta_factor_map = torch.stack(factor_maps).sum(dim=0)
    return delta_factor_map


class AdaptationEvaluator(KnockoutEvaluator):
    def __init__(self, model: MambaForCausalLM, tokenizer: AutoTokenizer, device: torch.device, feature_map: Dict[str, Iterable[int]], factor: Dict[str, float], feature_mask: Dict[str, int], show_progress: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.show_progress = show_progress
        self.feature_map = feature_map
        self.factor = factor
        self.feature_mask = feature_mask

    def setup_model(self, layers: Iterable[int]) -> None:
        for i in range(len(self.model.backbone.layers)):
            if i in layers:
                # "mixer of interest" - moi
                moi = self.model.backbone.layers[i].mixer

                feature_factor_map = build_delta_factor_map(self.feature_map, self.factor, moi.A_log.shape[0], i)
                feature_mask = build_delta_factor_map(self.feature_map, self.feature_mask, moi.A_log.shape[0], i)

                adapted_mixer = AdaptedMixer(moi, feature_factor_map, feature_mask)
                self.model.backbone.layers[i].mixer = adapted_mixer

    def knockout_eval(self, dataset: pd.DataFrame, layers: Iterable[int], knockout_mode: KnockoutMode) -> Tuple[pd.DataFrame, int]:
        acc = 0
        
        dataset['correct'] = False
        self.setup_model(layers)
                
        # Evaluate model
        pbar = tqdm(dataset.index, total=len(dataset), disable=not self.show_progress)
        for idx in pbar:
            # Get relevant data
            input = dataset.loc[idx, "prompt"]
            target = dataset.loc[idx, "attribute"]
            subj = dataset.loc[idx, "subject"]
            

            input_ids = self.tokenizer(input, return_tensors="pt")["input_ids"].to(self.device)
            
            out = self.model(input_ids)

            # get last decoded word
            decoded = self.tokenizer.decode(out.logits.argmax(dim=-1).squeeze())
            last_word = decoded.split(' ')[-1]

            correct = last_word == target[:len(last_word)]
            # Update performance
            acc += float(correct) / len(dataset)
            dataset.loc[idx, 'correct'] = correct

        return dataset, acc