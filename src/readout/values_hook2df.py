from typing import List
import pandas as pd
from .ssm_listener import SSMListenerHook
import torch


def values_hooks2df(hooks: List[SSMListenerHook]) -> pd.DataFrame:
    df = {'input': [], 'layer': [], 'matrix_type': [], 'T_1': [], 'T_2': [], 'value': [], 'DN': []}
    for hook in hooks:
        T = hook.metric_values['T']
        tril = torch.tril_indices(T, T, offset=-1)
        for key in hook.metric_values.keys():
            if key == 'T':
                continue
            value_matrix = hook.metric_values[key]
            for i in range(value_matrix.shape[-1]):
                for d in range(value_matrix.shape[1]):
                    df['input'].append(hook.input)
                    df['layer'].append(hook.layer)
                    df['matrix_type'].append(key)
                    # Assume that the value matrix has shape (B=1, n, 3)

                    df['T_1'].append(value_matrix[0, d, 0, i].detach().cpu().numpy())
                    df['T_2'].append(value_matrix[0, d, 1, i].detach().cpu().numpy())
                    df['DN'].append(d)
                    df['value'].append(value_matrix[0, d, 2, i].detach().cpu().numpy())
            
    df = pd.DataFrame(df)
    return df