from typing import Callable, Optional

import torch
from torch import Tensor, nn


class LLMEmbeddingInterefere(Callable):
    def __init__(self, layer: int | str | nn.Module, embedding_matrix: Tensor, k_closest: int = 1):
        self.layer = layer
        self.E = embedding_matrix
        self.k_closest = k_closest

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        # project to token space
        token_logits = out @ self.E.T

        # gather top k closest tokens
        top_k_logits = torch.topk(token_logits, self.k_closest, dim=-1).values
        thresholds = torch.min(top_k_logits, dim=-1, keepdim=True).values

        use_softmax = False
        if use_softmax:
            token_logits[token_logits < thresholds] = -float("inf")

            # normalize to get a distribution
            distribution = torch.softmax(token_logits, dim=-1)
        else:
            token_logits[token_logits < thresholds] = 0
            distribution = token_logits

        # project back to embedding space
        new_out = distribution @ self.E / torch.norm(self.E, dim=-1)
        return new_out

    def __call__(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        return self.hook(module, inp, out)
