import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wget
import pandas as pd
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from tqdm import tqdm
from pathlib import Path
from src.knockout import KnockoutMode, KnockoutTarget, SSMInterfereHook, choose_knockout_target, is_last_token_subj
from argparse import ArgumentParser
import numpy as np
from typing import Iterable


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, choices={'130M', '2.8B'}, default="130M")
    parser.add_argument("--interfere_mode", type=str, choices={str(mode).split('.')[1] for mode in KnockoutMode}, default="ZERO_ATTENTION")
    parser.add_argument("--interfere_target", type=str, choices=[str(target).split('.')[1] for target in KnockoutTarget] + [None], default=None)
    parser.add_argument("--drop_subj_last", action='store_true')
    parser.add_argument("--show_eval_progress", action='store_true')
    return parser.parse_args()


def knockout_eval(model: MambaForCausalLM, tokenizer: AutoTokenizer, knowns_df: pd.DataFrame, device: torch.device, interefere_mode: KnockoutMode, layer_indices: Iterable[int], knockout_target: KnockoutTarget, affected_target: KnockoutTarget, drop_subj_last: bool, show_progress: bool = False):
    acc = 0
    hooks = []
    handles = []

    # set up hooks
    for i in range(len(model.backbone.layers)):
        if i in layer_indices:
            # "mixer of interest" - moi
            moi = model.backbone.layers[i].mixer

            hooks.append(SSMInterfereHook(i, interefere_mode))
            
            handles.append(moi.register_forward_hook(hooks[-1]))

    # Evaluate model
    pbar = tqdm(knowns_df.index, total=len(knowns_df), disable=not show_progress)
    for idx in pbar:
        # Get relevant data
        input = knowns_df.loc[idx, "prompt"]
        target = knowns_df.loc[idx, "attribute"]
        subj = knowns_df.loc[idx, "subject"]

        if drop_subj_last:
            if is_last_token_subj(input, subj, tokenizer):
                continue

        # set subject token as knockout idx
        knockout_indices = choose_knockout_target(input, subj, tokenizer, knockout_target)
        affected_target_indices = choose_knockout_target(input, subj, tokenizer, affected_target)
        
        for hook in hooks:
            hook.knockout_indices = knockout_indices
            hook.affected_outputs = affected_target_indices

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


def main_binary_search(model_size: str = "2.8B", interefere_mode: KnockoutMode = KnockoutMode.ZERO_ATTENTION, interefere_target: KnockoutTarget = KnockoutTarget.ENTIRE_SUBJ, affected_outputs: KnockoutTarget = KnockoutTarget.LAST, drop_subj_last: bool = False, show_eval_progress: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path("known_1000.json").exists():
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json").set_index('known_id')

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.to(device)

    model.eval()

    performance = {'acc': [], 'layer': [], 'start_layer': [], 'end_layer': []}

    knowns_df['model_correct'] = False

    knockout_target_layers = list(range(len(model.backbone.layers)))
    
    acc = knockout_eval(model, tokenizer, knowns_df, device, interefere_mode, knockout_target_layers, interefere_target, affected_outputs, drop_subj_last, show_eval_progress)
    performance['layer'].append(str(knockout_target_layers))
    performance['start_layer'].append(min(knockout_target_layers))
    performance['end_layer'].append(max(knockout_target_layers))
    performance['acc'].append(acc)

    # log(n) binary search
    n = len(knockout_target_layers)
    log_n = np.ceil(np.log2(n))

    with torch.no_grad():
        for _ in tqdm(range(int(log_n)), desc='Binary search for optimal layer'):
            if (len(knockout_target_layers) // 2) < (len(knockout_target_layers) / 2):
                early = knockout_target_layers[:len(knockout_target_layers) // 2 + 1]
                late = knockout_target_layers[len(knockout_target_layers) // 2:]
            else:
                early = knockout_target_layers[:len(knockout_target_layers) // 2]
                late = knockout_target_layers[len(knockout_target_layers) // 2:]

            acc_early = knockout_eval(model, tokenizer, knowns_df, device, interefere_mode, early, interefere_target, drop_subj_last, show_eval_progress)
            performance['layer'].append(str(early))
            performance['acc'].append(acc_early)
            performance['start_layer'].append(min(early))
            performance['end_layer'].append(max(early))

            acc_late = knockout_eval(model, tokenizer, knowns_df, device, interefere_mode, late, interefere_target, drop_subj_last, show_eval_progress)
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


def main(model_size: str = "2.8B", interefere_mode: KnockoutMode = KnockoutMode.ZERO_ATTENTION, interefere_target: KnockoutTarget = KnockoutTarget.ENTIRE_SUBJ, affected_outputs: KnockoutTarget = KnockoutTarget.LAST, drop_subj_last: bool = False, show_eval_progress: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path("known_1000.json").exists():
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json").set_index('known_id')

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.to(device)

    model.eval()

    performance = {'acc': [], 'layer': []}

    knowns_df['model_correct'] = False

    with torch.no_grad():
        # evaluate every single layer
        for i in tqdm(range(len(model.backbone.layers)), desc="Iterating layers for knockout..."):
            acc = knockout_eval(model, tokenizer, knowns_df, device, interefere_mode, [i], interefere_target, affected_outputs, drop_subj_last, show_eval_progress)
            performance['layer'].append(i)
            performance['acc'].append(acc)
        
    df = pd.DataFrame(performance)
    
    return df


def get_last_token_stats(model_size: str = '130M'):
    if not Path("known_1000.json").exists():
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json").set_index('known_id')

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")

    pbar = tqdm(knowns_df.index, total=len(knowns_df), disable=False)
    stat = 0
    for idx in pbar:
        # Get relevant data
        input = knowns_df.loc[idx, "prompt"]
        target = knowns_df.loc[idx, "attribute"]
        subj = knowns_df.loc[idx, "subject"]

        val = is_last_token_subj(input, subj, tokenizer)

        stat += val / len(knowns_df)

    print(stat)


if __name__ == "__main__":
    args = get_args()
    get_last_token_stats(args.model_size)

    bin_search_dfs = []
    layer_dfs = []
    if args.interfere_target is not None:
        targets = [KnockoutTarget[args.interfere_target]]
    else:
        targets = KnockoutTarget

    affected_outputs = [KnockoutTarget.LAST, KnockoutTarget.ENTIRE_SUBJ]
    # targets = [KnockoutTarget.SUBJ_CONTEXT]
    # affected_outputs = [KnockoutTarget.ENTIRE_SUBJ]

    for target in targets:
        for output in affected_outputs:
            bin_search_dfs.append(main_binary_search(args.model_size, KnockoutMode[args.interfere_mode], target, output, args.drop_subj_last, args.show_eval_progress))
            bin_search_dfs[-1]['knockout_inputs'] = target
            bin_search_dfs[-1]['affected_outputs'] = output

            layer_dfs.append(main(args.model_size, KnockoutMode[args.interfere_mode], target, output, args.show_eval_progress))
            layer_dfs[-1]['knockout_inputs'] = target
            layer_dfs[-1]['affected_outputs'] = output
    
    df = pd.concat(bin_search_dfs)
    df.to_csv("ssm_interference_bin_search.csv")
    
    df = pd.concat(layer_dfs)
    df.to_csv("ssm_interference_layer_by_layer.csv")
    