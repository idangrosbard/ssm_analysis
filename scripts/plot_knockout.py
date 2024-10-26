import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


BASELINE_PERFORMANCE = 0.6964433416046368

def plot_performance(performance: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.02, subplot_titles=("KnockoutTarget.LAST", "KnockoutTarget.ENTIRE_SUBJ"))
    
    color = {KnockoutTarget.ENTIRE_SUBJ: 'red', KnockoutTarget.SUBJ_LAST: 'blue', KnockoutTarget.FIRST: 'green', KnockoutTarget.LAST: 'yellow', KnockoutTarget.RANDOM: 'black', KnockoutTarget.RANDOM_SPAN: 'black', KnockoutTarget.ALL_CONTEXT: 'black', KnockoutTarget.SUBJ_CONTEXT: 'purple'}
    color = {str(key): color[key] for key in color}
    
    for i, affected_output in enumerate(performance['affected_outputs'].unique()):
        for target in tqdm(performance['knockout_inputs'].unique()):
            curr = performance[(performance['knockout_inputs'] == target) &(performance['affected_outputs'] == affected_output)]
            first = True
            for layer in curr['layer'].unique():
                curr_layer = curr[curr['layer'] == layer]
                fig.add_trace(go.Scatter(x=curr_layer['x'], y=curr_layer['acc'], mode='lines+markers', line=dict(color=color[str(target)]), name=f"{str(target)}", legendgroup=str(target), showlegend=first), row = i+1, col=1)
                first = False
        
    return fig


def main():
    args = get_args()
    performance = pd.read_csv(args.binary_search_results)
    performance = performance.melt(id_vars=['layer', 'knockout_inputs', 'affected_outputs', 'acc'], value_vars=['start_layer', 'end_layer'], value_name='x')
    print(performance)
    fig = plot_performance(performance)
    fig.update_layout(title="SSM Interference", template='plotly_white')

    # save figure
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.add_hline(y=BASELINE_PERFORMANCE, line_dash="dot", line_color="black")
    fig.write_html(args.output)


if __name__ == "__main__":
    main()