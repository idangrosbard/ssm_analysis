import torch

from src.knockout.attention_knockout.knockout_scan import knockout_scan
from src.knockout.knockout_mode import KnockoutMode

SSM_D = 1  # SSM dimension
INTRM_D = 1  # Intermediate dimension
B_D = 1  # Batch dimension


def setup_binary_test(seq_len: int):
    """Setup a test case where inputs are powers of 2, making it easy to track information flow in binary."""
    ssm_state = torch.zeros((B_D, INTRM_D, seq_len, SSM_D))
    # Create inputs as powers of 2: [1, 2, 4, 8, 16, ...]
    inputs = torch.tensor([2**i for i in range(seq_len)], dtype=torch.float32)
    discrete_A = torch.ones((B_D, INTRM_D, seq_len, SSM_D))
    B = torch.ones((B_D, INTRM_D, seq_len, INTRM_D)) / SSM_D
    deltaB_u = B * inputs.view(B_D, 1, seq_len, 1)  # Normalize inputs
    C = torch.ones((B_D, seq_len, SSM_D))
    return ssm_state, discrete_A, deltaB_u, C


def test_binary_pattern():
    """Test knockout behavior using powers of 2 as inputs.
    This makes it easy to track which inputs contribute to each output in binary."""
    T = 5
    ssm_state, discrete_A, deltaB_u, C = setup_binary_test(T)

    # Test with multiple knockouts
    outputs = knockout_scan(
        seq_len=T,
        ssm_state=ssm_state,
        discrete_A=discrete_A,
        deltaB_u=deltaB_u,
        C=C,
        knocked_out_inputs=[0, 1, 2],  # Knockout first three positions
        affected_outputs=[4],  # Affect last position
        knockout_mode=KnockoutMode.ZERO_ATTENTION,
        dtype=torch.float32,
    )

    # Expected binary patterns based on notebook example
    expected_values = [1.0, 3.0, 7.0, 15.0, 24.0]  # These values show clear binary patterns

    for i, (output, expected) in enumerate(zip(outputs, expected_values)):
        assert torch.allclose(output, torch.tensor([[expected]]), rtol=1e-5), (
            f"Mismatch at position {i}: got {output.item()}, expected {expected}"
        )


def test_simple_scan():
    """Test the basic scan operation without knockouts."""
    T = 5
    ssm_state, discrete_A, deltaB_u, C = setup_binary_test(T)

    # Run without any knockouts
    outputs = knockout_scan(
        seq_len=T,
        ssm_state=ssm_state,
        discrete_A=discrete_A,
        deltaB_u=deltaB_u,
        C=C,
        knocked_out_inputs=[],
        affected_outputs=[],
        knockout_mode=KnockoutMode.ZERO_ATTENTION,
        dtype=torch.float32,
    )

    # Expected cumulative sums based on notebook example
    expected_cumsum = [1.0, 3.0, 7.0, 15.0, 31.0]

    for i, (output, expected) in enumerate(zip(outputs, expected_cumsum)):
        assert torch.allclose(output, torch.tensor([[expected]]), rtol=1e-5), (
            f"Mismatch at position {i}: got {output.item()}, expected {expected}"
        )
