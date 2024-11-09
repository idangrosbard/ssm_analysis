import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.knockout.attention_knockout import KnockoutTarget
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--binary_search_results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="ssm_interference.html")
    parser.add_argument("--layer_results", type=Path, required=False, default=None)
    parser.add_argument("--knockout_mode", type=str, choices={'attention', 'ssm'}, default='attention')
    return parser.parse_args()


BASELINE_PERFORMANCE = 0.6964433416046368

def rgb_to_str(rgb: tuple[int, int, int]) -> str:
    return f"rgb{rgb}"


def get_attention_style():
    RED = (204, 0, 0)
    BLUE = (0, 0, 204)
    PURPLE = (102, 0, 204)
    
    color = {KnockoutTarget.ENTIRE_SUBJ: rgb_to_str(PURPLE), KnockoutTarget.SUBJ_LAST: rgb_to_str(PURPLE), KnockoutTarget.FIRST: rgb_to_str(BLUE), KnockoutTarget.LAST: rgb_to_str(RED), KnockoutTarget.RANDOM: 'gray', KnockoutTarget.RANDOM_SPAN: 'gray', KnockoutTarget.ALL_CONTEXT: 'black', KnockoutTarget.SUBJ_CONTEXT: rgb_to_str(BLUE)}
    dash = {KnockoutTarget.ENTIRE_SUBJ: 'solid', KnockoutTarget.SUBJ_LAST: 'dot', KnockoutTarget.FIRST: 'dash', KnockoutTarget.LAST: 'dash', KnockoutTarget.RANDOM: 'dot', KnockoutTarget.RANDOM_SPAN: 'solid', KnockoutTarget.ALL_CONTEXT: 'solid', KnockoutTarget.SUBJ_CONTEXT: 'solid'}

    color = {str(key): color[key] for key in color}
    dash = {str(key): dash[key] for key in dash}

    return color, dash


def get_ssm_style():
    RED = (204, 0, 0)
    BLUE = (0, 0, 204)
    PURPLE = (102, 0, 204)
    
    color = {'min': rgb_to_str(BLUE), 'max': rgb_to_str(RED), 'mid': rgb_to_str(PURPLE)}
    dash = {'min': 'solid', 'max': 'solid', 'mid': 'solid'}

    color = {str(key): color[key] for key in color}
    dash = {str(key): dash[key] for key in dash}

    return color, dash


def plot_bin_search(performance: pd.DataFrame, fig: go.Figure, row: int) -> go.Figure:

    interesting_tests = [KnockoutTarget.LAST, KnockoutTarget.ENTIRE_SUBJ, KnockoutTarget.SUBJ_LAST, KnockoutTarget.RANDOM_SPAN, KnockoutTarget.SUBJ_CONTEXT]
    interesting_tests = {str(test) for test in interesting_tests}

    color, dash = get_attention_style()
    
    
    for target in tqdm(performance['knockout_inputs'].unique()):
        if str(target) not in interesting_tests:
            continue
        curr = performance[(performance['knockout_inputs'] == target)]
        first = True
        for layer in curr['layer'].unique():
            curr_layer = curr[curr['layer'] == layer]
            fig.add_trace(go.Scatter(x=curr_layer['x'], y=curr_layer['acc'], mode='lines+markers', line=dict(color=color[str(target)]), name=f"{str(target)}", legendgroup=str(target), showlegend=first, line_dash=dash[str(target)]), row=row+1, col=1)
            first = False
        
    return fig


def plot_ssm_bin_search(performance: pd.DataFrame, fig: go.Figure) -> go.Figure:

    color, dash = get_ssm_style()
    
    for i, category in enumerate(performance['category'].unique()):
    
        curr = performance[(performance['category'] == category)]
        first = True
        for layer in curr['layer'].unique():
            curr_layer = curr[curr['layer'] == layer]
            fig.add_trace(go.Scatter(x=curr_layer['x'], y=curr_layer['acc'], mode='lines+markers', line=dict(color=color[str(category)]), name=f"{str(category)}", legendgroup=str(category), showlegend=first, line_dash=dash[str(category)]))
            first = False
        
    return fig


def plot_layer_by_layer(performance: pd.DataFrame, fig: go.Figure) -> go.Figure:

    interesting_tests = [KnockoutTarget.LAST, KnockoutTarget.ENTIRE_SUBJ, KnockoutTarget.SUBJ_LAST, KnockoutTarget.RANDOM_SPAN, KnockoutTarget.SUBJ_CONTEXT]
    interesting_tests = {str(test) for test in interesting_tests}

    color, dash = get_attention_style()
    
    for i, affected_output in enumerate(performance['affected_outputs'].unique()):
        for target in tqdm(performance['knockout_inputs'].unique()):
            if str(target) not in interesting_tests:
                continue
            curr = performance[(performance['knockout_inputs'] == target) &(performance['affected_outputs'] == affected_output)]
            first = True
            
            fig.add_trace(go.Scatter(x=curr['layer'], y=curr['acc'], mode='lines+markers', line=dict(color=color[str(target)]), name=f"{str(target)}", legendgroup=str(target), showlegend=first, line_dash=dash[str(target)]))
            first = False
        
    return fig



def main():
    args = get_args()
    performance = pd.read_csv(args.binary_search_results)
    if args.knockout_mode == 'attention':
        performance = performance.melt(id_vars=['layer', 'knockout_inputs', 'affected_outputs', 'acc'], value_vars=['start_layer', 'end_layer'], value_name='x')

    elif args.knockout_mode == 'ssm':
        performance = performance.melt(id_vars=['layer', 'norm', 'category', 'acc'], value_vars=['start_layer', 'end_layer'], value_name='x')
    
    
    if args.knockout_mode == 'attention':
        fig = make_subplots(rows=performance['affected_outputs'].nunique(), cols=1, subplot_titles=performance['affected_outputs'].unique())

        for i, affected_output in enumerate(performance['affected_outputs'].unique()):
            output_performance = performance[performance['affected_outputs'] == affected_output]
            fig = plot_bin_search(output_performance, fig, i)
    elif args.knockout_mode == 'ssm':
        fig = go.Figure()
        fig = plot_ssm_bin_search(performance, fig)
    
    fig.update_layout(title="SSM Interference", template='plotly_white', xaxis_title="Layer", yaxis_title="Accuracy", legend_title="Knockout Target", font=dict(size=18), height=1600, width=1000)
    # set minimum y value as 0
    fig.update_yaxes(range=[0, 0.8])
    fig.update_xaxes(range=[0-0.05, 63+0.05])

    # save figure
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.add_hline(y=BASELINE_PERFORMANCE, line_dash="dot", line_color="black")
    fig.write_html(args.output)


if __name__ == "__main__":
    main()