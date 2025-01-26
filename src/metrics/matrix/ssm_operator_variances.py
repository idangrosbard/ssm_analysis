import numpy as np
import torch
from sklearn.decomposition import PCA

from .matrix_metric import MatrixMetric


class SSMOperatorVariances(MatrixMetric):
    def calc(self, **kwargs) -> np.ndarray:
        mat = kwargs["mat"]
        assert type(mat) == torch.Tensor
        assert len(mat.shape) in [4, 5]

        if len(mat.shape) == 4:
            B, D, T, T = mat.shape
            N = 1
        elif len(mat.shape) == 5:
            B, D, T, N, T = mat.shape

        mat = mat.reshape(B, D * N, T, T)

        # Set nans on upper triangle
        nan_mat = torch.ones_like(mat)
        nan_mat[:] = torch.tensor([(float("nan"))])
        mat = torch.tril(mat) + torch.triu(nan_mat, 1)

        # remove upper triangle
        idx = torch.tril_indices(T, T)
        mat = mat[:, :, idx[0], idx[1]]

        # reshape to 3D
        n_keep = (T * T - T) // 2 + T
        mat = mat.reshape(B, D * N, n_keep)
        mat = mat.cpu().detach().numpy()

        variances_ratio = []
        # Calculate variances for each sample in batch
        for i in range(B):
            pca = PCA()
            pca.fit(mat[i])
            variances_ratio.append(pca.explained_variance_ratio_)
        variances_ratio = np.array(variances_ratio)
        return variances_ratio
