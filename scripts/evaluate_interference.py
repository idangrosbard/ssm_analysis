import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wget
import pandas as pd
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from tqdm import tqdm
from pathlib import Path
from src.hooks import SSMInterfereHook
from src.updates_ssm_ops import KnockoutMode
import plotly.express as px
from argparse import ArgumentParser
from typing import Tuple
import random
import numpy as np
import plotly.graph_objects as go
from src.updates_ssm_ops import KnockoutMode, KnockoutTarget


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, choices={'130M', '2.8B'}, default="130M")
    parser.add_argument("--interfere_mode", type=str, choices={str(mode) for mode in KnockoutMode}, default="ZERO_ATTENTION")
    parser.add_argument("--interfere_target", type=str, choices={str(target) for target in KnockoutTarget}, default="ENTIRE_SUBJ")
    return parser.parse_args()


def get_subj_idx(input: str, subj: str, tokenizer: AutoTokenizer) -> Tuple[int,int]:
    prefix = input.split(subj)[0]
    sent2subj = prefix
    
    if prefix == "":
        sent2subj = subj
    else:
        sent2subj = prefix + ' ' + subj

    sent2subj_tokens = tokenizer(sent2subj)["input_ids"]
    prefix_tokens = tokenizer(prefix)["input_ids"]
    return (len(prefix_tokens), len(sent2subj_tokens))


def choose_knockout_target(input: str, subj: str, tokenizer: AutoTokenizer, target: KnockoutTarget) -> Tuple[int,int]:
    if target == KnockoutTarget.ENTIRE_SUBJ:
        return get_subj_idx(input, subj, tokenizer)
    elif target == KnockoutTarget.SUBJ_LAST:
        first, last = get_subj_idx(input, subj, tokenizer)
        return (last, last + 1)
    elif target == KnockoutTarget.FIRST:
        return (0, 1)
    elif target == KnockoutTarget.LAST:
        return (len(tokenizer(input)["input_ids"]) - 1, len(tokenizer(input)["input_ids"]))
    elif target == KnockoutTarget.RANDOM:
        # TODO remove the subject from the possible choices
        first = random.randint(0, len(tokenizer(input)["input_ids"]))
        return (first, first + 1)
    elif target == KnockoutTarget.RANDOM_SPAN:
        first = random.randint(0, len(tokenizer(input)["input_ids"]))
        last = random.randint(0, len(tokenizer(input)["input_ids"]))
        return min(first, last), max(first, last) + 1
    

def plot_performance(performance: pd.DataFrame):
    
    fig = go.Figure()
    color = {KnockoutTarget.ENTIRE_SUBJ: 'red', KnockoutTarget.SUBJ_LAST: 'blue', KnockoutTarget.FIRST: 'green', KnockoutTarget.LAST: 'yellow', KnockoutTarget.RANDOM: 'purple', KnockoutTarget.RANDOM_SPAN: 'orange'}
    for target in performance['target'].unique():
        curr = performance[performance['target'] == target]
        fig.add_trace(go.Scatter(x=curr['layer'], y=curr['acc'], mode='lines+markers', color=color[target], name=str(target)))
    
    fig.write_html("ssm_interference_subject.html")


def knockout_eval(model, tokenizer, knowns_df, device, interefere_mode: KnockoutMode, layer_indices, interfere_target):
    acc = 0
    hooks = []
    handles = []

    # set up hooks
    for i in range(len(model.backbone.layers)):
        if i in layer_indices:
            moi = model.backbone.layers[i].mixer

            hooks.append(SSMInterfereHook(i, interefere_mode))
            
            handles.append(moi.register_forward_hook(hooks[-1]))

    # Evaluate model
    pbar = tqdm(knowns_df.index, total=len(knowns_df), disable=True)
    for idx in pbar:
        # Get relevant data
        input = knowns_df.loc[idx, "prompt"]
        target = knowns_df.loc[idx, "attribute"]
        subj = knowns_df.loc[idx, "subject"]

        # set subject token as knockout idx
        out = get_subj_idx(input, subj, tokenizer)
        out = choose_knockout_target(input, subj, tokenizer, interfere_target)
        start_idx, end_idx = out
        for hook in hooks:
            hook.knockout_start_idx = start_idx
            hook.knockout_end_idx = end_idx

        input_ids = tokenizer(input, return_tensors="pt")["input_ids"].to(device)
        out = model(input_ids)

        # get last decoded word
        decoded = tokenizer.decode(out.logits.argmax(dim=-1).squeeze())
        last_word = decoded.split(' ')[-1]

        # Update performance
        acc += float(last_word == target[:len(last_word)]) / len(knowns_df)
    
    # remove hooks
    for handle in handles:
        handle.remove()

    return acc


def main_binary_search(model_size: str = "2.8B", interefere_mode: KnockoutMode = KnockoutMode.ZERO_ATTENTION, interefere_target: KnockoutTarget = KnockoutTarget.ENTIRE_SUBJ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path("known_1000.json").exists():
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json").set_index('known_id')
    print(knowns_df)

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.to(device)

    model.eval()

    performance = {'acc': [], 'layer': [], 'start_layer': [], 'end_layer': []}

    knowns_df['model_correct'] = False

    knockout_target_layers = list(range(len(model.backbone.layers)))
    
    acc = knockout_eval(model, tokenizer, knowns_df, device, interefere_mode, knockout_target_layers, interefere_target)
    performance['layer'].append(str(knockout_target_layers))
    performance['start_layer'].append(min(knockout_target_layers))
    performance['end_layer'].append(max(knockout_target_layers))
    performance['acc'].append(acc)

    # log(n) binary search
    n = len(knockout_target_layers)
    log_n = np.log2(n)

    with torch.no_grad():
        for _ in tqdm(range(int(log_n)), desc='Binary search for optimal layer'):
            early = knockout_target_layers[:len(knockout_target_layers) // 2]
            late = knockout_target_layers[len(knockout_target_layers) // 2:]
            acc_early = knockout_eval(model, tokenizer, knowns_df, device, interefere_mode, early, interefere_target)
            performance['layer'].append(str(early))
            performance['acc'].append(acc_early)
            performance['start_layer'].append(min(early))
            performance['end_layer'].append(max(early))
            acc_late = knockout_eval(model, tokenizer, knowns_df, device, interefere_mode, late, interefere_target)
            performance['layer'].append(str(late))
            performance['acc'].append(acc_late)
            performance['start_layer'].append(min(late))
            performance['end_layer'].append(max(late))
            
            if acc_early < acc_late:
                knockout_target_layers = early
            else:
                knockout_target_layers = late
    
    df = pd.DataFrame(performance)
    
    return df


def main(model_size: str = "2.8B", interefere_mode: KnockoutMode = KnockoutMode.ZERO_ATTENTION, interefere_target: KnockoutTarget = KnockoutTarget.ENTIRE_SUBJ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path("known_1000.json").exists():
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json").set_index('known_id')
    print(knowns_df)

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.to(device)

    model.eval()

    performance = {'acc': [], 'layer': []}

    knowns_df['model_correct'] = False

    with torch.no_grad():
        # evaluate every single layer
        for i in tqdm(range(len(model.backbone.layers)), desc="Iterating layers for knockout..."):
            continue
            acc = knockout_eval(model, tokenizer, knowns_df, device, interefere_mode, [i], interefere_target)
            performance['layer'].append(i)
            performance['acc'].append(acc)
        
        # evaluate all layers at once
        acc = knockout_eval(model, tokenizer, knowns_df, device, interefere_mode, list(range(len(model.backbone.layers))), interefere_target)
        performance['layer'].append(len(model.backbone.layers))
        performance['acc'].append(acc)

    
    df = pd.DataFrame(performance)
    df.to_csv("ssm_interference_subject.csv")
    long = df.melt(id_vars=['layer', 'acc'], var_name='mode', value_name='x')
    fig = go.Figure()
    for layer in long['layer'].unique():
        curr = long[long['layer'] == layer]
        fig.add_trace(go.Scatter(x=curr['x'], y=curr['acc'], mode='lines+markers', name=f'Layer {layer}', color='red'))
    fig.write_html("ssm_interference_subject.html")


if __name__ == "__main__":
    args = get_args()
    dfs = []
    for target in KnockoutTarget:
        dfs.append(main_binary_search(args.model_size, KnockoutMode[args.interfere_mode], target))
        dfs[-1]['target'] = str(target)
    
    df = pd.concat(dfs)
    plot_performance(df)
    df.to_csv("ssm_interference_subject.csv")

    # main(args.model_size, KnockoutMode[args.interfere_mode], KnockoutTarget[args.interfere_target])
    