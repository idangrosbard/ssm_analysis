from typing import Callable, Optional

from torch import Tensor, nn

from .mamba_mixer_increase_delta import slow_forward_for_ssm_manipulation


class IncreaseDeltaHook(Callable):
    def __init__(self, layer: int | str | nn.Module, feature_map: Tensor, factor: float):
        self.counter = 0
        self.layer = layer
        self.feature_map = feature_map
        self.factor = factor
        self.factored_tokens = None

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        """
        module - the actual layer
        inp - previous layer ssm state (in mamba 1 - this is X)
        out - the current layer output ssm state (in mamba 1 \ 2 - this is Y)
        """
        # TODO make the knockout more efficient - maybe replace the module.forward with a custom forward
        curr_out = slow_forward_for_ssm_manipulation(
            module,
            inp[0],
            feature_mask=self.feature_map,
            factored_tokens=self.factored_tokens,
            factor=self.factor,
        )
        return curr_out

    def __call__(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        return self.hook(module, inp, out)

    def __str__(self):
        return f"SSMInterfereHook for layer {self.layer} with knockout type {self.knockout_type}, knockout indices {self.knockout_indices}, affected outputs {self.affected_outputs}"
