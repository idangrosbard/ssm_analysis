from ..updates_ssm_ops import slow_forward_for_ssm_materializing_knockout
from ..updates_ssm_ops import KnockoutMode
from ..metrics.ssm.ssm_metric import SSMMetric
from torch import nn, Tensor
from typing import Optional, Callable


class SSMInterfereHook(Callable):
    def __init__(self, layer: int | str | nn.Module, knockout_type: KnockoutMode, knockout_idx: int):
        self.counter = 0
        self.layer = layer
        self.knockout_type = knockout_type
        self.knockout_idx = knockout_idx

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        curr_out = slow_forward_for_ssm_materializing_knockout(module, inp[0], knockout_idx=self.knockout_idx, knockout_mode=self.knockout_type)
        return curr_out
        
    def __call__(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        return self.hook(module, inp, out)