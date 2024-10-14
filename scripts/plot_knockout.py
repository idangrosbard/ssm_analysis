import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import plotly.graph_objects as go
from src.knockout import KnockoutTarget
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--binary_search_results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="ssm_interference.html")
    parser.add_argument("--layer_results", type=Path, required=False, default=None)
    return parser.parse_args()


def plot_performance(performance: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    color = {KnockoutTarget.ENTIRE_SUBJ: 'red', KnockoutTarget.SUBJ_LAST: 'blue', KnockoutTarget.FIRST: 'green', KnockoutTarget.LAST: 'yellow', KnockoutTarget.RANDOM: 'purple', KnockoutTarget.RANDOM_SPAN: 'orange'}
    color = {str(key): color[key] for key in color}
    for target in tqdm(performance['target'].unique()):
        curr = performance[performance['target'] == target]
        first = True
        for layer in curr['layer'].unique():
            curr_layer = curr[curr['layer'] == layer]
            fig.add_trace(go.Scatter(x=curr_layer['x'], y=curr_layer['acc'], mode='lines+markers', line=dict(color=color[str(target)]), name=f"{str(target)}", legendgroup=str(target), showlegend=first))
            first = False
        
    return fig
    


def main():
    args = get_args()
    performance = pd.read_csv(args.binary_search_results)
    performance = performance.melt(id_vars=['layer', 'target', 'acc'], value_vars=['start_layer', 'end_layer'], value_name='x')
    fig = plot_performance(performance)
    fig.update_layout(title="SSM Interference", template='plotly_white')

    # save figure
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output)


if __name__ == "__main__":
    main()