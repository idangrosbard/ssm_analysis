from typing import List
import pandas as pd
from .ssm_listener import SSMListenerHook


def summarized_hooks2df(hooks: List[SSMListenerHook]) -> pd.DataFrame:
    df = {'input': [], 'layer': [], 'matrix_type': [], 'entropy': []}
    for hook in hooks:
        for key in hook.metric_values.keys():
            if key == 'T':
                continue
            df['input'].append(hook.input)
            df['layer'].append(hook.layer)
            df['matrix_type'].append(key)
            df['entropy'].append(hook.metric_values[key])

    df = pd.DataFrame(df)
    return df