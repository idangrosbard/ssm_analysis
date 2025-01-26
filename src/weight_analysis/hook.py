from typing import Callable, Optional

from torch import Tensor, nn

from .mamba_mixer_context_split import slow_forward_for_ssm_materializing_listener


class Hook(Callable):
    def __init__(self):
        self.context_state = None
        self.independent_state = None

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        self.counter += 1

        curr_out = slow_forward_for_ssm_materializing_listener(module, inp[0])
        state, indep_states, context_state = curr_out

        self.context_state = context_state
        self.independent_state = indep_states

    def __call__(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        return self.hook(module, inp, out)
