import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, MambaModel
from src.hooks import SSMListenerHook, hooks2df
from src.metrics import SSMOperatorVariances, SSMOperatorEntropy, AllSSMEntropies
import torch
import plotly.express as px
import pandas as pd


def _plot_series(df: pd.DataFrame):
    print(df)
    return px.line(data_frame=df, x='layer', y='entropy', color='matrix_type', title='Entropy per layer')




def main():
    input = "Hey how are you doing?"
    # input = "Toko Yasuda produces the most amazing music on the"
    input = "The mother tongue of Go Hyeon-jeong is"
    hooks = []
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-790m-hf")
    model = MambaModel.from_pretrained("state-spaces/mamba-790m-hf")
    for i in range(len(model.layers)):
        moi = model.layers[i].mixer

        counter = SSMListenerHook(input, i, AllSSMEntropies(SSMOperatorEntropy(SSMOperatorVariances())))
        hooks.append(counter)
        
        handle = moi.register_forward_hook(counter.hook)

        input_ids = tokenizer(input, return_tensors="pt")["input_ids"]

        out = model(input_ids)
        handle.remove()
    
    df = hooks2df(hooks)

    _plot_series(df).write_html("ssm_entropies.html")



if __name__ == '__main__':
    main()