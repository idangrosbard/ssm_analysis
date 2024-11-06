import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wget
import pandas as pd
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from tqdm import tqdm
from pathlib import Path
from src.embeddings import LLMEmbeddingInterefere
from src.datasets.download_dataset import load_knowns_pd
from argparse import ArgumentParser
import numpy as np
import plotly.graph_objects as go
from typing import Iterable


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, choices={'130M', '2.8B'}, default="130M")
    parser.add_argument("--power", type=int, default=10)
    return parser.parse_args()


def tokenize_eval(model: MambaForCausalLM, tokenizer: AutoTokenizer, knowns_df: pd.DataFrame, device: torch.device, layer_indices: Iterable[int], n_tokens: int = 1):
    acc = 0
    hooks = []
    handles = []

    # set up hooks
    E = model.backbone.embeddings.weight
    for i in range(len(model.backbone.layers)):
        if i in layer_indices:
            # "module of interest" - moi (mixer or layer?)
            moi = model.backbone.layers[i].mixer

            hooks.append(LLMEmbeddingInterefere(i, E, n_tokens))
            
            handles.append(moi.register_forward_hook(hooks[-1]))

    # Evaluate model
    pbar = tqdm(knowns_df.index, total=len(knowns_df), disable=True)
    for idx in pbar:
        # Get relevant data
        input = knowns_df.loc[idx, "prompt"]
        target = knowns_df.loc[idx, "attribute"]
        subj = knowns_df.loc[idx, "subject"]

        # set subject token as knockout idx
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


def log_scale_k_closest_search(model_size: str, k: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knowns_df = load_knowns_pd()

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.to(device)

    model.eval()

    performance = {'acc': [], 'k': []}

    knowns_df['model_correct'] = False

    knockout_target_layers = list(range(len(model.backbone.layers)))
    
    # log(n) binary search
    log_k = np.ceil(np.log2(k)) + 1
    
    pbar = tqdm(range(int(log_k)), desc='Binary search for optimal layer')
    with torch.no_grad():
        for _ in pbar:
            acc = tokenize_eval(model, tokenizer, knowns_df, device, knockout_target_layers, k)
            performance['acc'].append(acc)
            performance['k'].append(k)

            pbar.set_description(f'k: {k}, acc: {acc}')

            k = k // 2
            
    df = pd.DataFrame(performance)
    
    return df


def main_binary_search(model_size: str = "2.8B", n_tokens: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knowns_df = load_knowns_pd()

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.to(device)

    model.eval()

    performance = {'acc': [], 'layer': [], 'start_layer': [], 'end_layer': []}

    knowns_df['model_correct'] = False

    knockout_target_layers = list(range(len(model.backbone.layers)))
    
    acc = tokenize_eval(model, tokenizer, knowns_df, device, knockout_target_layers)
    performance['layer'].append(str(knockout_target_layers))
    performance['start_layer'].append(min(knockout_target_layers))
    performance['end_layer'].append(max(knockout_target_layers))
    performance['acc'].append(acc)

    # log(n) binary search
    n = len(knockout_target_layers)
    log_n = np.ceil(np.log2(n))
    
    pbar = tqdm(range(int(log_n)), desc='Binary search for optimal layer')
    with torch.no_grad():
        for _ in pbar:
            pbar.set_description(f'{str(knockout_target_layers)}: {acc}')
            if (len(knockout_target_layers) // 2) < (len(knockout_target_layers) / 2):
                early = knockout_target_layers[:len(knockout_target_layers) // 2 + 1]
                late = knockout_target_layers[len(knockout_target_layers) // 2:]
            else:
                early = knockout_target_layers[:len(knockout_target_layers) // 2]
                late = knockout_target_layers[len(knockout_target_layers) // 2:]

            acc_early = tokenize_eval(model, tokenizer, knowns_df, device, early)
            performance['layer'].append(str(early))
            performance['acc'].append(acc_early)
            performance['start_layer'].append(min(early))
            performance['end_layer'].append(max(early))

            acc_late = tokenize_eval(model, tokenizer, knowns_df, device, late)
            performance['layer'].append(str(late))
            performance['acc'].append(acc_late)
            performance['start_layer'].append(min(late))
            performance['end_layer'].append(max(late))
            
            if acc_early < acc_late:
                knockout_target_layers = early
            else:
                knockout_target_layers = late

            acc = min(acc_early, acc_late)
    
    df = pd.DataFrame(performance)
    
    return df


def main(model_size: str = "2.8B"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knowns_df = load_knowns_pd()

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
            acc = tokenize_eval(model, tokenizer, knowns_df, device, interefere_mode, [i], interefere_target, drop_subj_last)
            acc = tokenize_eval(model, tokenizer, knowns_df, device, [i])
            performance['layer'].append(i)
            performance['acc'].append(acc)
        
        # evaluate all layers at once
        acc = tokenize_eval(model, tokenizer, knowns_df, device, list(range(len(model.backbone.layers))))
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

    df = log_scale_k_closest_search(args.model_size, 2 ** args.power)
    
    df.to_csv("ssm_tokenize.csv")
    # main(args.model_size, KnockoutMode[args.interfere_mode], KnockoutTarget[args.interfere_target])
    