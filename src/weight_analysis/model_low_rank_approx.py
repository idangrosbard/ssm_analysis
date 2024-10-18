from transformers import MambaModel
from typing import Iterable
from .rank import create_low_rank_approx
from torch import nn



def get_low_rank_model(model: MambaModel, rank: int | Iterable[int], use_max_vals: bool = True) -> MambaModel:
    n_layers = len(model.layers)
    if type(rank) == int:
        rank = [rank] * n_layers
    elif type(rank) == Iterable:
        assert len(rank) == n_layers, f"number of ranks ({len(rank)}) must match number of layers ({n_layers})"
    else:
        raise ValueError("rank must be an int or an Iterable of ints")
    
    for layer, r in zip(model.layers, rank):
        w = layer.mixer.out_proj.weight.detach()
        w = create_low_rank_approx(w, r, use_max_vals)
        layer.mixer.out_proj.weight = nn.Parameter(w)
    
    return model
