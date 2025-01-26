from typing import Tuple

import torch


def materialize_ssm_transition(A: torch.Tensor) -> torch.Tensor:
    batch = A.shape[0]
    D = A.shape[1]
    T = A.shape[2]
    N = A.shape[3]
    A = A.transpose(-1, -2).repeat(1, 1, 1, T).reshape(batch, D, N, T, T).transpose(-1, -2)
    A = torch.tril(A) + torch.triu(torch.ones_like(A), 1)
    A_cumprod = torch.cumprod(A, axis=-2)

    transition_mat = A_cumprod.transpose(-2, -3)

    return transition_mat


def materialize_ssm_bc(B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    bc = torch.einsum("btn, bdqn -> bdtq", C, B)

    return bc


def materialize_ssm_attention(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, return_transition: bool
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    transition_mat = materialize_ssm_transition(A)

    AB = transition_mat * B.unsqueeze(-1)

    out = torch.einsum("btn, bdtnq -> bdtq", C, AB)

    if return_transition:
        return out, transition_mat

    return out
