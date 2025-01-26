from torch import Tensor, matmul, zeros_like
from .. import KnockoutMode
from typing import List, Iterable
import torch


def knockout_scan(seq_len: int, ssm_state: Tensor, discrete_A: Tensor, deltaB_u: Tensor, C: Tensor, knocked_out_inputs: Iterable[int], affected_outputs: Iterable[int], knockout_mode: KnockoutMode, dtype) -> List[Tensor]:
    knockout_state = zeros_like(ssm_state)
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
        if i not in knocked_out_inputs:
            knockout_state = discrete_A[:, :, i, :] * knockout_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
        elif i in knocked_out_inputs:
            if knockout_mode == KnockoutMode.ZERO_ATTENTION:
                knockout_state = discrete_A[:, :, i, :] * knockout_state
            elif knockout_mode == KnockoutMode.ZERO_DELTA:
                knockout_state = knockout_state
        if i in affected_outputs:
            scan_output = matmul(knockout_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        else:
            scan_output = matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        scan_outputs.append(scan_output[:, :, 0])

    return scan_outputs


def materialize_ssm_transition(A: torch.Tensor) -> torch.Tensor:
    batch = A.shape[0]
    D = A.shape[1]
    T = A.shape[2]
    N = A.shape[3]
    A = A.transpose(-1,-2).repeat(1,1,1,T).reshape(batch,D,N,T,T).transpose(-1,-2)
    A = torch.tril(A) + torch.triu(torch.ones_like(A),1)
    A_cumprod = torch.cumprod(A, dim=-2)

    transition_mat = A_cumprod.transpose(-2,-3)

    return transition_mat


def materialize_ssm_attention(A: Tensor, B: Tensor, C: Tensor, return_transition: bool) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: 
    transition_mat = materialize_ssm_transition(A)

    AB = (transition_mat * B.unsqueeze(-1))

    out = torch.einsum('btn, bdtnq -> bdtq', C, AB)
    
    if return_transition:
        return out, transition_mat
    
    return out


def knockout_matrix(seq_len: int, discrete_A: Tensor, discrete_B: Tensor, u: Tensor, C: Tensor, knocked_out_inputs: Iterable[int], affected_outputs: Iterable[int], dtype) -> List[Tensor]:
    attn = materialize_ssm_attention(discrete_A, discrete_B, C, False)
    for i, j in zip(affected_outputs, knocked_out_inputs):
        attn[:, :, i, j] = 0
    outputs = (attn @ u).squeeze(-1)
    return outputs


def ignore_context_knockout_scan(seq_len: int, ssm_state: Tensor, discrete_A: Tensor, deltaB_u: Tensor, C: Tensor, knockout_start_idx: int, knockout_end_idx: int, dtype) -> List[Tensor]:
    knockout_state = zeros_like(ssm_state)
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
        # TODO: Test this to see if it works, (prime numbers)
        if (i >= knockout_start_idx) and (i < knockout_end_idx):
            knockout_state = discrete_A[:, :, i, :] * knockout_state + deltaB_u[:, :, i, :]  # [batch, intermediade_size, ssm_state]
            scan_output = matmul(knockout_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        else:
            scan_output = matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        scan_outputs.append(scan_output[:, :, 0])

    return scan_outputs


