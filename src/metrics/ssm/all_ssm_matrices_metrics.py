from .ssm_metric import SSMMetric
from ..matrix.matrix_metric import MatrixMetric
import numpy as np
from typing import Dict
from torch import Tensor
from .materialize import materialize_ssm_attention, materialize_ssm_transition, materialize_ssm_bc


class AllSSMMatricesMetrics(SSMMetric):
    def __init__(self, matrix_metric: MatrixMetric):
        self.matrix_metric = matrix_metric

    def calc(self, A: Tensor, B: Tensor, C: Tensor) -> Dict[str, np.ndarray | int]:
        attn, transition = materialize_ssm_attention(A, B, C, return_transition=True)
        bc = materialize_ssm_bc(B, C)
        if len(transition.shape) == 4:
            B, D, T, T = transition.shape
            N = 1
        elif len(transition.shape) == 5:
            B, D, T, N, T = transition.shape
        attn_metric = self.matrix_metric.calc(mat=attn)
        transition_metric = self.matrix_metric.calc(mat=transition)
        bc_metric = self.matrix_metric.calc(mat=bc)
        return {'attn': attn_metric, 'transition': transition_metric, 'bc': bc_metric, 'T': T}
    