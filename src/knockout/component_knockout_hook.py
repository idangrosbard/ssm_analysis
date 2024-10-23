from .mamba_mixer_context_split import slow_forward_for_context_indep_split
from .knockout_mode import KnockoutMode
from torch import nn, Tensor
from typing import Optional, Callable


class ComponentKnockoutHook(Callable):
    def __init__(self, layer: int | str | nn.Module, knockout_type: KnockoutMode):
        self.counter = 0
        self.layer = layer
        assert knockout_type in {KnockoutMode.IGNORE_CONTEXT, KnockoutMode.ONLY_CONTEXT, KnockoutMode.IGNORE_LAYER}
        self.knockout_type = knockout_type
        self.knockout_indices = {}
        self.affected_outputs = {}

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        '''
        module - the actual layer
        inp - previous layer ssm state (in mamba 1 - this is X)
        out - the current layer output ssm state (in mamba 1 \ 2 - this is Y)
        '''
        # TODO make the knockout more efficient - maybe replace the module.forward with a custom forward
        contextualized_states, indep_states, context_states = slow_forward_for_context_indep_split(module, inp[0])
        if self.knockout_type == KnockoutMode.IGNORE_CONTEXT:
            return indep_states
        elif self.knockout_type == KnockoutMode.ONLY_CONTEXT:
            return context_states
        elif self.knockout_type == KnockoutMode.IGNORE_LAYER:
            return inp[0]
        else:
            raise ValueError(f"Unknown knockout type: {self.knockout_type}")
        
    def __call__(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        return self.hook(module, inp, out)