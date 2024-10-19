from transformers import MambaModel
from typing import Iterable
from .rank import create_low_rank_approx
from torch import nn


def setup_rank(rank: int, model_rank: int) -> int:
    if rank < 0:
        rank = model_rank + rank
    return rank


def setup_low_ranks(rank: int | Iterable[int], model_rank: int, n_layers: int) -> Iterable[int]:
    if type(rank) == int:
        rank = setup_rank(rank, model_rank)
        rank = [rank] * n_layers
    elif type(rank) == Iterable:
        for i, r in enumerate(rank):
            rank[i] = setup_rank(r, model_rank)
        assert len(rank) == n_layers, f"number of ranks ({len(rank)}) must match number of layers ({n_layers})"
    else:
        raise ValueError("rank must be an int or an Iterable of ints")
    return rank


def get_low_rank_model(model: MambaModel, rank: int | Iterable[int], use_max_vals: bool = True) -> MambaModel:

    rank = setup_low_ranks(rank, model_rank=model.layers[0].mixer.out_proj.weight.shape[0], n_layers=len(model.layers))
    
    for layer, r in zip(model.layers, rank):
        w = layer.mixer.out_proj.weight.detach()
        w = create_low_rank_approx(w, r, use_max_vals)
        layer.mixer.out_proj.weight = nn.Parameter(w)
    
    return model
