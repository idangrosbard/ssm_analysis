from typing import Callable, Optional

from torch import Tensor, nn

from src.knockout.knockout_mode import KnockoutMode

from .mamba_mixer_knockout import slow_forward_for_ssm_materializing_knockout


class SSMInterfereHook(Callable):
    def __init__(self, layer: int | str | nn.Module, knockout_type: KnockoutMode):
        self.counter = 0
        self.layer = layer
        self.knockout_type = knockout_type
        self.knockout_indices = {}
        self.affected_outputs = {}

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        """
        module - the actual layer
        inp - previous layer ssm state (in mamba 1 - this is X)
        out - the current layer output ssm state (in mamba 1 \ 2 - this is Y)
        """
        # TODO make the knockout more efficient - maybe replace the module.forward with a custom forward
        curr_out = slow_forward_for_ssm_materializing_knockout(
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
        return f"SSMInterfereHook for layer {self.layer} with knockout type {self.knockout_type}, knockout indices {self.knockout_indices}, affected outputs {self.affected_outputs}"
