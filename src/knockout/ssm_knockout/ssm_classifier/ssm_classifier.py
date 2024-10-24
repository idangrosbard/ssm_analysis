from transformers import MambaModel
from typing import Dict, Iterable
import torch


class SSMClassifier(object):
    def classify_layer(self, layer: torch.nn.Module) -> Dict[str, Iterable[int]]:
        raise NotImplementedError
    
    def classify_model(self, model: MambaModel) -> Dict[str, Iterable[Iterable[int]]]:
        classifications = {}
        for layer in model.layers:
            curr = self.classify_layer(layer.mixer)
            for category in curr:
                if category not in classifications:
                    classifications[category] = []
                classifications[category].append(curr[category])
        
        return classifications