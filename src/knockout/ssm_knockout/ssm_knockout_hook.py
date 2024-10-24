from .mamba_mixer_ssm_knockout import slow_forward_for_ssm_knockout
from torch import nn, Tensor, device
import torch
from typing import Optional, Callable, Iterable


def indices2khot(indices: Iterable[int], len: int, flip: bool = True) -> torch.Tensor:
    if type(indices) is not torch.Tensor:
        indices = torch.tensor(indices, dtype=torch.long)
    one_hots = torch.nn.functional.one_hot(indices, len)
    k_hot = one_hots.sum(dim=0)
    if flip:
        k_hot = 1 - k_hot
    return k_hot


class SSMKnockoutHook(Callable):
    def __init__(self, layer: int | str | nn.Module, knockout_indices: Iterable[int], device: torch.device, d: int):
        self.counter = 0
        self.layer = layer
        knockout_indices = indices2khot(knockout_indices, d).to(device)
        self.knockout_indices = knockout_indices

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        '''
        module - the actual layer
        inp - previous layer ssm state (in mamba 1 - this is X)
        out - the current layer output ssm state (in mamba 1 \ 2 - this is Y)
        '''
        # TODO make the knockout more efficient - maybe replace the module.forward with a custom forward
        contextualized_states = slow_forward_for_ssm_knockout(module, inp[0], mask=self.knockout_indices)
        return contextualized_states
        
    def __call__(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        return self.hook(module, inp, out)