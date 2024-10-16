from torch import Tensor, matmul, zeros_like
from .knockout_mode import KnockoutMode
from typing import List


def terminal_token_knockout_scan(seq_len: int, ssm_state: Tensor, discrete_A: Tensor, deltaB_u: Tensor, C: Tensor, knockout_start_idx: int, knockout_end_idx: int, knockout_mode: KnockoutMode, dtype) -> List[Tensor]:
    final_state = zeros_like(ssm_state)
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
        # TODO: Test this to see if it works, (prime numbers)
        if (i < knockout_start_idx) or (i >= knockout_end_idx):
            final_state = discrete_A[:, :, i, :] * final_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
        elif (i >= knockout_start_idx) and (i < knockout_end_idx):
            if knockout_mode == KnockoutMode.ZERO_ATTENTION:
                final_state = discrete_A[:, :, i, :] * final_state
            elif knockout_mode == KnockoutMode.ZERO_DELTA:
                final_state = final_state
        scan_output = matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        scan_outputs.append(scan_output[:, :, 0])

    final_state = matmul(final_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
    scan_output[-1] = final_state
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


