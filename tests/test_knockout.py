import torch

from src.knockout.attention_knockout.knockout_scan import knockout_scan
from src.knockout.knockout_mode import KnockoutMode


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def get_primes(n):
    primes = [1]
    i = 1
    while len(primes) < n:
        i += 1
        if is_prime(i):
            primes.append(i)
    return primes


def setup_test(seq_len: int):
    seq_len = 10
    primes = get_primes(10)
    ssm_state = torch.zeros((1, 1, 1, 1))
    discrete_A = torch.Tensor(primes).view(1, 1, 10, 1)
    deltaB_u = torch.Tensor(primes).view(1, 1, 10, 1)
    C = torch.ones((1, 10, 1))
    knockout_start_idx = 3
    knockout_end_idx = 6
    knockout_mode = KnockoutMode.ZERO_ATTENTION
    dtype = torch.float32
    outputs = knockout_scan(
        seq_len,
        ssm_state,
        discrete_A,
        deltaB_u,
        C,
        knockout_start_idx,
        knockout_end_idx,
        knockout_mode,
        dtype,
    )
    return outputs


def test_correct_non_last():
    outputs = setup_test(10)
    primes = torch.Tensor(get_primes(10))
    A = torch.cumprod(primes, dim=0)
    expected = cumprod * primes

    for i in range(9):
        assert outputs[i].item() == expected[i].item()
