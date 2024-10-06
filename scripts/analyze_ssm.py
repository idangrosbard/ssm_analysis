import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, MambaModel
from src.hooks import SSMListenerHook, summarized_hooks2df, values_hooks2df
from src.metrics import SSMOperatorVariances, SSMOperatorEntropy, AllSSMMatricesMetrics, SSMOperatorValueMap
import torch
import plotly.express as px
import pandas as pd
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--metric", type=str, choices={'entropy', 'values'}, default="entropy")
    return parser.parse_args()


def _plot_summarized(df: pd.DataFrame):
    print(df)
    return px.line(data_frame=df, x='layer', y='entropy', color='matrix_type', title='Entropy per layer')

def _plot_values(df: pd.DataFrame):
    print(df)
    for mat_type in df['matrix_type'].unique():
        yield (mat_type, px.scatter_3d(data_frame=df[df['matrix_type'] == mat_type], x='T_1', y='T_2', z='value', color='DN', title='matrix_value'))


def main():
    args = get_args()
    input = "Hey how are you doing?"
    # input = "Toko Yasuda produces the most amazing music on the"
    input = "The mother tongue of Go Hyeon-jeong is"
    hooks = []
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    model = MambaModel.from_pretrained("state-spaces/mamba-130m-hf")
    model.eval()
    with torch.no_grad():
        for i in range(len(model.layers)):
            moi = model.layers[i].mixer

            if args.metric == "entropy":
                metric = SSMOperatorEntropy(SSMOperatorVariances())
            elif args.metric == "values":
                metric = SSMOperatorValueMap()

            counter = SSMListenerHook(input, i, AllSSMMatricesMetrics(metric))
            hooks.append(counter)
            
            handle = moi.register_forward_hook(counter.hook)

            input_ids = tokenizer(input, return_tensors="pt")["input_ids"]

            out = model(input_ids)
            handle.remove()
            if args.metric == "values":
                df = values_hooks2df(hooks)
                plots = _plot_values(df)
                for mat_type, fig in plots:
                    fig.write_html(f"ssm_values_layer_{i}_mat_{mat_type}.html")
                hooks = []
        

        else:
            df = summarized_hooks2df(hooks)
            _plot_summarized(df).write_html("ssm_entropies.html")

    
if __name__ == '__main__':
    main()