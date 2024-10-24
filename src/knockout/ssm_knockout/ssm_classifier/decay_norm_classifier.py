from .ssm_classifier import SSMClassifier
# from transformers import
from typing import Dict, Iterable
import torch


class DecayNormClassifier(SSMClassifier):
    def __init__(self, ratio: float = 1/3, n_groups: int = 3, norm: int | float = 1):
        self.ratio = ratio
        self.n_groups = n_groups
        self.norm = norm

    @staticmethod
    def _materialize_decay_matrices(A_log: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.exp(A_log))

    def classify_layer(self, layer: torch.nn.Module) -> Dict[str, Iterable[int]]:
        decay_matrices = DecayNormClassifier._materialize_decay_matrices(layer.A_log)
        n_ssms = decay_matrices.shape[0]

        # get the norms
        if self.norm == float('inf'):
            norms = torch.max(decay_matrices, dim=1)
        else:
            norms = torch.norm(decay_matrices, p=self.norm, dim=1)
        
        # sort and divide to groups
        sorted_indices = torch.argsort(norms, descending=True)
        group_size = n_ssms // self.n_groups
        groups = {}
        if self.n_groups == 3:
            labels = ['max', 'mid', 'min']
        else:
            labels = [f"group_{i}" for i in range(self.n_groups)]
        for i, label in enumerate(labels):
            groups[label] = sorted_indices[i*group_size:(i+1)*group_size]
        return groups
    
        

