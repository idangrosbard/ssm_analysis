from typing import Iterable, Optional

from torch import Tensor, nn

from src.knockout.attention_knockout.mamba_mixer_knockout import slow_forward_for_ssm_materializing_knockout
from src.knockout.attention_knockout.mamba_mixer_knockout_falcon import (
    slow_forward_for_ssm_materializing_knockout_falcon,
)
from src.knockout.knockout_mode import KnockoutMode


class SSMInterfereHook:
    def __init__(self, layer: int | str | nn.Module, knockout_type: KnockoutMode, is_falcon: bool):
        self.counter = 0
        self.layer = layer
        self.is_falcon = is_falcon
        self.knockout_type = knockout_type
        self.knockout_indices: Iterable[int] = []
        self.affected_outputs: Iterable[int] = []

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        """
        module - the actual layer
        inp - previous layer ssm state (in mamba 1 - this is X)
        out - the current layer output ssm state (in mamba 1 / 2 - this is Y)
        """
        # TODO make the knockout more efficient - maybe replace the module.forward with a custom forward
        slow_forward = (
            slow_forward_for_ssm_materializing_knockout_falcon
            if self.is_falcon
            else slow_forward_for_ssm_materializing_knockout
        )
        curr_out = slow_forward(
            module,
            inp[0],
            knockout_indices=self.knockout_indices,
            affected_outputs=self.affected_outputs,
            knockout_mode=self.knockout_type,
        )
        return curr_out

    def __call__(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        return self.hook(module, inp, out)

    def __str__(self):
        return (
            f"SSMInterfereHook for layer {self.layer} "
            f"with knockout type {self.knockout_type}, "
            f"knockout indices {self.knockout_indices}, "
            f"affected outputs {self.affected_outputs}"
        )
