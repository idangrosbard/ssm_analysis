import torch

def get_singular_values(w: torch.Tensor) -> torch.Tensor:
    u, s, v = torch.svd(w)
    return s


def get_topk_singular_vectors(w: torch.Tensor, k: int) -> torch.Tensor:
    u, s, v = torch.svd(w)
    return v[:k]