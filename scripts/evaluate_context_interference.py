import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from pathlib import Path
from src.knockout import KnockoutMode, KnockoutTarget, AttentionKnockoutEvaluator, LayerKnockoutEvaluator, KnockoutEvaluator, is_last_token_subj
from argparse import ArgumentParser
import numpy as np
from src.utils import load_knowns, setup_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, choices={'130M', '2.8B'}, default="130M")
    parser.add_argument("--interfere_mode", type=str, choices={str(mode).split('.')[1] for mode in KnockoutMode}, default="ZERO_ATTENTION")
    parser.add_argument("--interfere_target", type=str, choices=[str(target).split('.')[1] for target in KnockoutTarget] + [None], default=None)
    parser.add_argument("--drop_subj_last", action='store_true')
    parser.add_argument("--show_eval_progress", action='store_true')
    parser.add_argument("--output_dir", type=Path, default=Path("resources"))
    return parser.parse_args()


def binary_search(evaluator: KnockoutEvaluator, dataset: pd.DataFrame, knockout_mode: KnockoutMode) -> pd.DataFrame:
    performance = {'acc': [], 'layer': [], 'start_layer': [], 'end_layer': []}

    knockout_target_layers = list(range(len(evaluator.model.backbone.layers)))
    
    acc = evaluator.knockout_eval(dataset, knockout_target_layers, knockout_mode)
    
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

            _, acc_early = evaluator.knockout_eval(dataset, early, knockout_mode)
            performance['layer'].append(str(early))
            performance['acc'].append(acc_early)
            performance['start_layer'].append(min(early))
            performance['end_layer'].append(max(early))

            _, acc_late = evaluator.knockout_eval(dataset, late, knockout_mode)
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


def layer_by_layer(evaluator: KnockoutEvaluator, dataset: pd.DataFrame, knockout_mode: KnockoutMode) -> pd.DataFrame:
    performance = {'acc': [], 'layer': []}

    with torch.no_grad():
        # evaluate every single layer
        for i in tqdm(range(len(evaluator.model.backbone.layers)), desc="Iterating layers for knockout..."):
            _, acc = evaluator.knockout_eval(dataset, [i], knockout_mode)
            performance['layer'].append(i)
            performance['acc'].append(acc)
        
    df = pd.DataFrame(performance)
    
    return df


def get_last_token_stats(model_size: str = '130M'):
    knowns_df = load_knowns()

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


def main() -> None:
    args = get_args()
    get_last_token_stats(args.model_size)

    tokenizer, model, device = setup_model(args.model_size)
    knowns_df = load_knowns()

    
    # If we do attention knockout:
    if KnockoutMode[args.interfere_mode] in {KnockoutMode.ZERO_ATTENTION, KnockoutMode.ZERO_DELTA}:
        evaluator = AttentionKnockoutEvaluator(model, tokenizer, device, KnockoutTarget[args.interfere_target], KnockoutTarget[args.affected_outputs], args.drop_subj_last, args.show_eval_progress)

        bin_search_dfs = []
        layer_dfs = []

        if args.interfere_target is not None:
            targets = [KnockoutTarget[args.interfere_target]]
        else:
            targets = KnockoutTarget
        affected_outputs = [KnockoutTarget.LAST, KnockoutTarget.ENTIRE_SUBJ]
        
        for target in targets:
            for output in affected_outputs:
                bin_search_dfs.append(binary_search(evaluator, knowns_df, KnockoutMode[args.interfere_mode]))
                bin_search_dfs[-1]['knockout_inputs'] = target
                bin_search_dfs[-1]['affected_outputs'] = output

                layer_dfs.append(layer_by_layer(evaluator, knowns_df, KnockoutMode[args.interfere_mode]))
                layer_dfs[-1]['knockout_inputs'] = target
                layer_dfs[-1]['affected_outputs'] = output

        bin_search_df = pd.concat(bin_search_dfs)
        layer_df = pd.concat(layer_dfs)

    # If we skip entire layer \ component
    elif KnockoutMode[args.interfere_mode] in {KnockoutMode.IGNORE_CONTEXT, KnockoutMode.IGNORE_LAYER, KnockoutMode.ONLY_CONTEXT}:
        evaluator = LayerKnockoutEvaluator(model, tokenizer, device, args.show_eval_progress)
        bin_search_df = binary_search(evaluator, knowns_df, KnockoutMode[args.interfere_mode])
        layer_df = layer_by_layer(evaluator, knowns_df, KnockoutMode[args.interfere_mode])
    else:
        raise ValueError(f"Unknown knockout mode: {args.interfere_mode}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bin_search_df.to_csv(args.output_dir / f"{args.interfere_mode}_bin_search.csv")
    layer_df.to_csv(args.output_dir / f"{args.interfere_mode}_layer_by_layer.csv")
    

if __name__ == "__main__":
    main()