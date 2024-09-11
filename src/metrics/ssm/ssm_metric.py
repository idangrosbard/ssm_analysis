import numpy as np
from typing import Dict
from torch import Tensor


class SSMMetric(object):
    """
    Base class for all metrics
    """

    def calc(self, A: Tensor, B: Tensor, C: Tensor) -> Dict[str, np.ndarray]:
        raise NotImplementedError