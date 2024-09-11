import numpy as np



class MatrixMetric(object):
    def calc(self, **kwargs) -> np.ndarray:
        raise NotImplementedError
        