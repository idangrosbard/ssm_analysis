import torch

from src.knockout.attention_knockout.knockout_scan import knockout_scan
from src.knockout.knockout_mode import KnockoutMode


def setup_test(seq_len: int, primes: torch.Tensor):
    ssm_state = torch.zeros((1, 1, 1, 1))
    discrete_A = primes.view(1, 1, seq_len, 1)
    deltaB_u = primes.view(1, 1, seq_len, 1)
    C = torch.ones((1, seq_len, 1))
    knockout_start_idx = 3
    knockout_end_idx = 6
    knockout_mode = KnockoutMode.ZERO_ATTENTION
    outputs = knockout_scan(
        seq_len=seq_len,
        ssm_state=ssm_state,
        discrete_A=discrete_A,
        deltaB_u=deltaB_u,
        C=C,
        knocked_out_inputs=[knockout_start_idx],
        affected_outputs=[knockout_end_idx],
        knockout_mode=knockout_mode,
        dtype=torch.float32,
    )
    return outputs


def test_correct_non_last():
    N = 10
    PRIMES = torch.Tensor([1, 2, 3, 5, 7, 11, 13, 17, 19, 23])
    assert len(PRIMES) == N

    outputs = setup_test(N, PRIMES)
    primes = PRIMES
    A = torch.cumprod(primes, dim=0)
    expected = A * primes

    for i in range(9):
        assert outputs[i].item() == expected[i].item()
