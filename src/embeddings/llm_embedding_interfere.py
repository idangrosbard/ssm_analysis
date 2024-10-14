from torch import nn, Tensor
import torch
from typing import Optional, Callable, Iterable


class LLMEmbeddingInterefere(Callable):
    def __init__(self, layer: int | str | nn.Module, embedding_matrix: Tensor, k_closest: int = 1):
        self.layer = layer
        self.E = embedding_matrix
        self.k_closest = k_closest


    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        # project to token space
        token_logits = out @ self.E.T
        # print(token_logits.shape)

        # gather top k closest tokens
        top_k_logits = torch.topk(token_logits, self.k_closest, dim=-1).values
        thresholds = torch.min(top_k_logits[:, -1], dim=-1).values

        token_logits[token_logits < thresholds] = 0

        # normalize to get a distribution
        distribution = torch.softmax(token_logits, dim=-1)
        # print(distribution.shape)

        # project back to embedding space
        return distribution @ self.E

        
    def __call__(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        return self.hook(module, inp, out)