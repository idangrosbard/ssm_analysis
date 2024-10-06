from ..updates_ssm_ops.mamba_mixer_listener import slow_forward_for_ssm_materializing
from ..metrics.ssm.ssm_metric import SSMMetric
from torch import nn, Tensor
from typing import Optional


class SSMListenerHook():
    def __init__(self, input: str, layer: int | str | nn.Module, metric: SSMMetric):
        self.counter = 0
        self.input = input
        self.layer = layer
        self.metric = metric
        self.metric_values = None

    def hook(self, module: nn.Module, inp: Tensor, out: Tensor) -> Optional[Tensor]:
        self.counter += 1

        curr_out = slow_forward_for_ssm_materializing(module, inp[0])
        state, A, B, C, discrete_time_step = curr_out

        self.metric_values = self.metric.calc(A, B, C)
