from .matrix_metric import MatrixMetric
import numpy as np
from scipy.stats import entropy


class SSMOperatorEntropy(MatrixMetric):
    def __init__(self, variance_calculation: MatrixMetric):
        self.variance_calculation = variance_calculation

    def calc(self, **kwargs) -> np.ndarray:
        mat = kwargs['mat']
        variances = self.variance_calculation.calc(mat=mat)
        return entropy(variances.flatten(), axis=-1)