from .mamba_mixer_knockout import slow_forward_for_ssm_materializing_knockout
from .knockout_mode import KnockoutMode
from torch import nn, Tensor
from typing import Optional, Callable


class SSMInterfereHook(Callable):
    def __init__(self, layer: int | str | nn.Module, knockout_type: KnockoutMode):
        self.counter = 0
        self.layer = layer
        self.knockout_type = knockout_type
        self.knockout_start_idx = -1
        self.knockout_end_idx = -1

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        '''
        module - the actual layer
        inp - previous layer ssm state (in mamba 1 - this is X)
        out - the current layer output ssm state (in mamba 1 \ 2 - this is Y)
        '''
        # TODO make the knockout more efficient - maybe replace the module.forward with a custom forward
        curr_out = slow_forward_for_ssm_materializing_knockout(module, inp[0], knockout_start_idx=self.knockout_start_idx, knockout_end_idx=self.knockout_end_idx, knockout_mode=self.knockout_type)
        return curr_out
        
    def __call__(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        return self.hook(module, inp, out)