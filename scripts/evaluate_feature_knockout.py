from dataclasses import dataclass
import sys
import os

is_nir = os.getenv('USER') == 'nirendy'
if is_nir:
    import pyrallis
    from src.utils.slurm import submit_job
else:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.consts import PATHS

import pandas as pd
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from tqdm import tqdm
from pathlib import Path
from src.knockout import KnockoutMode, KnockoutEvaluator
from src.knockout.ssm_knockout import SSMKnockoutEvaluator
from src.knockout.ssm_knockout.ssm_classifier import DecayNormClassifier
from argparse import ArgumentParser, Namespace
import numpy as np
from src.utils.setup_models import setup_mamba_model
from typing import Optional, Iterable


@dataclass
class Args:
    model_size: str = "2.8B"
    interfere_mode: str = "INCREASE_DELTA"
    drop_subj_last: bool = False
    show_eval_progress: bool = False
    output_dir: Path = PATHS.RESULTS_DIR / 'evaluate_context_interference'
    layer_checkpoint: Optional[Path] = None
    bin_search_checkpoint: Optional[Path] = None
    ignore_layer_by_layer: bool = False
    norm: str = '1'
    early_layers_ssm_knockout: bool = False
    affected_output: str = 'all'
    delta_factor_root: float = 0.9
    delta_start_layer: int = 40
    delta_end_layer: int = 48
    non_selective_ssm: bool = False
    increase_delta_target: str = "LAST"
    with_slurm:bool = False
    split_name:str='train1'

if not is_nir:
    def get_args():
        parser = ArgumentParser()
        parser.add_argument("--model_size", type=str, choices={'130M','370M', '790M', '1.4B', '2.8B'}, default="130M")
        parser.add_argument("--show_eval_progress", action='store_true')
        parser.add_argument("--output_dir", type=Path, default=Path("resources"))
        parser.add_argument("--layer_checkpoint", type=Path, default=None)
        parser.add_argument('--norm', type=str, default='1', choices=['1', 'inf'])
        parser.add_argument('--affected_output', type=str, choices={'last', 'subj', 'all'}, default='all')
        parser.add_argument('--layer_window_length', type=int, default=9)
        
        return parser.parse_args()


def binary_search(evaluator: KnockoutEvaluator, dataset: pd.DataFrame, knockout_mode: KnockoutMode, start_layers: Optional[Iterable[int]] = None) -> pd.DataFrame:
    if start_layers is None:
        knockout_target_layers = list(range(len(evaluator.model.backbone.layers)))
    else:
        knockout_target_layers = start_layers
    
    _, acc = evaluator.knockout_eval(dataset, knockout_target_layers, knockout_mode)
    
    performance = {'acc': [], 'layer': [], 'start_layer': [], 'end_layer': []}
    performance['layer'].append(str(knockout_target_layers))
    performance['start_layer'].append(min(knockout_target_layers))
    performance['end_layer'].append(max(knockout_target_layers))
    performance['acc'].append(acc)
    print(acc)

    # log(n) binary search
    n = len(knockout_target_layers)
    log_n = np.ceil(np.log2(n))

    with torch.no_grad():
        pbar = tqdm(range(int(log_n)), desc='Binary search for optimal layer')
        for _ in pbar:
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
            pbar.set_description(f"Binary search for optimal layer. Curr acc: {min(acc_early, acc_late)}")
    
    df = pd.DataFrame(performance)
    
    return df


def layer_by_layer(evaluator: KnockoutEvaluator, dataset: pd.DataFrame, knockout_mode: KnockoutMode, n_layers: int) -> pd.DataFrame:
    performance = {'acc': [], 'layer': []}
    n = len(evaluator.model.backbone.layers)

    with torch.no_grad():
        # evaluate every single layer
        for i in tqdm(range(n), desc="Iterating layers for knockout..."):
            _, acc = evaluator.knockout_eval(dataset, [j for j in range(i, min(n, i+n_layers))], knockout_mode)
            performance['layer'].append(i)
            performance['acc'].append(acc)
        
    df = pd.DataFrame(performance)
    
    return df



def ssm_knockout_evaluate(args: Namespace, model: MambaForCausalLM, tokenizer: AutoTokenizer, device: torch.device, knowns_df: pd.DataFrame, norm: int | float, n_layers):
    ssm_classifier = DecayNormClassifier()
    out_df = None
    layer_df = None
    ssm_classifier.norm = norm

    categorized_ssms = ssm_classifier.classify_model(model.backbone)
    for category in categorized_ssms:
        evaluator = SSMKnockoutEvaluator(model, tokenizer, device, categorized_ssms[category], False)
        curr = layer_by_layer(evaluator, knowns_df, KnockoutMode[args.interfere_mode], n_layers)
        curr['category'] = category
        curr['norm'] = norm
        out_df = pd.concat([out_df, curr])

        out_fname = args.output_dir / f"{args.interfere_mode}_{args.model_size}_norm_{norm}_bin_search.csv"
        if out_fname.exists():
            os.remove(out_fname)
        out_df.to_csv(out_fname)

        
    out_df.to_csv(args.output_dir / f"{args.interfere_mode}_{args.model_size}_norm_{norm}_output.csv")


def get_checkpoint(pth: Optional[Path]) -> Optional[pd.DataFrame]:
    if pth is not None:
        return pd.read_csv(pth)
    return None


def main_local(args:Args) -> None:
    print(args)
    model, tokenizer, device = setup_mamba_model(args.model_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bin_search_checkpoint = get_checkpoint(args.bin_search_checkpoint)
    layer_checkpoint = get_checkpoint(args.layer_checkpoint)
    print(bin_search_checkpoint)
    print(layer_checkpoint)
    
    # If we do SSM knockout
    # knowns_df = pd.DataFrame(load_dataset(DatasetArgs(name=DATASETS.KNOWN_1000, splits=[args.split_name])))
    knowns_df = pd.read_parquet("./entire_results_attention.parquet")

    # drop the first character in the attribute string
    # knowns_df['attribute'] = knowns_df['attribute'].apply(lambda x: x[1:])
    if args.norm == 'inf':
        norm = float('inf')
    else:
        norm = int(args.norm)
    
    ssm_knockout_evaluate(args, model, tokenizer, device, knowns_df, norm=norm, ignore_layer_by_layer=args.ignore_layer_by_layer, n_layers=args.layer_window_length)
    

if is_nir:
    def get_experiment_configs():
        """Generate configurations for different experiment variations."""
        base_config = {
            'model_size': '2.8B',
            'output_dir': '',
        }

        # Standard configurations
        configs = [
            {**base_config, 'interfere_mode': 'ZERO_ATTENTION'},
            {**base_config, 'interfere_mode': 'IGNORE_SSM'},
            {**base_config, 'interfere_mode': 'IGNORE_SSM', 'early_layers_ssm_knockout': True},
        ]

        # Delta configurations
        delta_variations = [
            {'delta_factor_root': 0.5, 'delta_start_layer': 40, 'delta_end_layer': 48},
            {'delta_factor_root': 1.5, 'delta_start_layer': 40, 'delta_end_layer': 48},
            {'delta_factor_root': 0.5, 'delta_start_layer': 56, 'delta_end_layer': 64},
            {'delta_factor_root': 1.5, 'delta_start_layer': 56, 'delta_end_layer': 64},
        ]

        # Add delta configurations
        for delta_config in delta_variations:
            configs.append({
                **base_config,
                'interfere_mode': 'INCREASE_DELTA',
                'increase_delta_target': 'LAST',
                **delta_config
            })

        return configs

    @pyrallis.wrap()
    def main(args: Args):
        if args.with_slurm:
            args.model_size = '2.8B'
            
            # gpu_type = "titan_xp-studentrun"
            # gpu_type = "titan_xp-studentbatch"
            gpu_type = "titan_xp-studentkillable"
            # gpu_type = "a100"
            
            job_name1 = f"evaluate_context_interference_{args.model_size}"
            args.output_dir = args.output_dir / args.model_size / 'split'

            for i in [
                1,
                # 2,
                ]:
                args.output_dir = args.output_dir.parent / f"split{i}"
                args.split_name = f"train{i}"
                job_name2 = job_name1 + f"_split={args.split_name}"

                for interfere_mode in [
                    # 'ZERO_ATTENTION',
                    # 'IGNORE_SSM', 
                    'INCREASE_DELTA',
                    ]:
                    args.interfere_mode = interfere_mode
                    job_name3 = job_name2 + f"_{interfere_mode}"
                    
                    mods:list[dict] = [{}]
                    
                    if interfere_mode == 'INCREASE_DELTA':
                        mods = [
                            {'delta_factor_root': 0.5, 'delta_start_layer': 40, 'delta_end_layer': 48, 'increase_delta_target': 'LAST'},
                            {'delta_factor_root': 1.5, 'delta_start_layer': 40, 'delta_end_layer': 48, 'increase_delta_target': 'LAST'},
                            {'delta_factor_root': 0.5, 'delta_start_layer': 56, 'delta_end_layer': 64, 'increase_delta_target': 'LAST'},
                            {'delta_factor_root': 1.5, 'delta_start_layer': 56, 'delta_end_layer': 64, 'increase_delta_target': 'LAST'},
                        ]
                        short_cuts = {
                            'delta_factor_root': 'dfr',
                            'delta_start_layer': 'dsl',
                            'delta_end_layer': 'del',
                            'increase_delta_target': 'idt'
                        }
                    if interfere_mode == 'IGNORE_SSM':
                        mods = [
                            {'early_layers_ssm_knockout': False},
                            {'early_layers_ssm_knockout': True},
                        ]
                        
                        short_cuts = {
                            'early_layers_ssm_knockout': 'elsk',
                        }
                        

                    # Update args with config
                    for mod in mods:
                        prev_vals = {}
                        job_name = job_name3
                        for key in mod:
                            prev_vals[key] = getattr(args, key)
                            setattr(args, key, mod[key])
                            job_name += f"_{short_cuts[key]}={mod[key]}"
                            
                        job = submit_job(
                            main_local,
                            args,
                            log_folder=str(PATHS.SLURM_DIR / job_name1 / job_name / "%j"),
                            job_name=job_name,
                            gpu_type=gpu_type,
                            slurm_gpus_per_node=2,
                        )

                        print(f"{job}: {job_name}")
                        # Restore args
                        for key in mod:
                            setattr(args, key, prev_vals[key])
        else:
            main_local(args)

if __name__ == "__main__":
    if is_nir:
        main()
    else:
        args = get_args()
        main_local(args)
