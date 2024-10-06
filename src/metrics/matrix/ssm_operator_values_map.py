import torch
import numpy as np
from .matrix_metric import MatrixMetric


class SSMOperatorValueMap(MatrixMetric):
    def calc(self, **kwargs) -> np.ndarray:
        mat = kwargs['mat']
        assert type(mat) == torch.Tensor
        assert len(mat.shape) in [4, 5]
        
        if len(mat.shape) == 4:
            B, D, T, T = mat.shape
            N = 1
        elif len(mat.shape) == 5:
            B, D, T, N, T = mat.shape

        mat = mat.reshape(B, D*N, T, T)

        # Set nans on upper triangle
        nan_mat = torch.ones_like(mat)
        nan_mat[:] = torch.tensor([(float('nan'))])
        mat = torch.tril(mat) + torch.triu(nan_mat, 1)

        # remove upper triangle
        idx = torch.tril_indices(T,T)
        mat = mat[:, :, idx[0], idx[1]]
        print(mat)
        batch_idx = idx.repeat(B, D*N, 1, 1)
        print(batch_idx)

        # reshape to 3D
        n_keep = (T * T - T) // 2 + T
        mat = mat.reshape(B, D*N, n_keep)
        mat = mat.unsqueeze(-2)
        batch_all = torch.concat((batch_idx, mat), dim=-2)
        print(batch_all)
        
        return batch_all