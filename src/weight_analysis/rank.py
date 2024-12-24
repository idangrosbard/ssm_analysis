import pandas as pd
import torch


def get_singular_values(w: torch.Tensor) -> torch.Tensor:
    u, s, v = torch.svd(w)
    return s


def get_topk_singular_vectors(w: torch.Tensor, k: int, top: bool = True) -> torch.Tensor:
    u, s, v = torch.svd(w)
    if top:
        return v[:k]
    else:
        return v[-k:]


def low_rank_approx_error_df(s: torch.Tensor) -> pd.DataFrame:
    reversed = s.flip(dims=[0])
    d = reversed.shape[0]
    cumsum = reversed.cumsum(dim=0)
    spectral_error = cumsum.flip(dims=[0])

    return pd.DataFrame(
        {
            "rank": (torch.arange(d) + 1).numpy(),
            "spectral_error": spectral_error.numpy(),
        }
    )


def create_low_rank_approx(w: torch.Tensor, k: int, use_max_vals: bool = True) -> torch.Tensor:
    u, s, v = torch.svd(w)
    if use_max_vals:
        s[k:] = 0
    else:
        s[:-k] = 0
    return u @ torch.diag(s) @ v.T


# def greatest_diff_vecs(w: torch.Tensor, w_tag: torch.Tensor, k: int, top: bool = True) -> torch.Tensor:
#     w_diff = w - w_tag
#     norm = torch.norm(w_diff, dim=-1)
#     indices = torch.topk(norm, k, dim=-1, largest=top).indices
#     return w[indices]
