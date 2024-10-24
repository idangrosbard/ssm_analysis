from torch import Tensor, matmul, zeros_like
from .. import KnockoutMode
from typing import List, Iterable


def knockout_scan(seq_len: int, ssm_state: Tensor, discrete_A: Tensor, deltaB_u: Tensor, C: Tensor, knocked_out_inputs: Iterable[int], affected_outputs: Iterable[int], knockout_mode: KnockoutMode, dtype) -> List[Tensor]:
    knockout_state = zeros_like(ssm_state)
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
        # TODO: Test this to see if it works, (prime numbers)
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


