import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from src.datasets.known_1000.download_dataset import load_knowns
from src.knockout.adapted_model import AdaptationEvaluator
from src.knockout.ssm_knockout.ssm_classifier import SSMClassifier, DecayNormClassifier, SSMClassifierStub
from argparse import ArgumentParser, Namespace
from src.utils import setup_model
from src.knockout.knockout_mode import KnockoutMode



def increase_delta_evaluate(args: Namespace, model: MambaForCausalLM, tokenizer: AutoTokenizer, device: torch.device, knowns_df: pd.DataFrame, root_factor: float, start_layer: int, end_layer: int, non_selective_ssm: bool):
    if args.model_size == '130M':
        layers_of_interest = [18, 19, 20, 21]
    else:
        layers_of_interest = [40, 41, 42, 43, 44, 45, 46, 47]
        layers_of_interest = sorted([63, 62, 61, 60, 59, 58, 57, 56])
        layers_of_interest = list(range(start_layer, end_layer))
    if non_selective_ssm:
        layer_classification = SSMClassifierStub().classify_model(model.backbone)
    else:
        layer_classification = DecayNormClassifier(norm=1).classify_model(model.backbone)

    performance = {'acc': [], 'layers': [], 'factor': [], 'category': []}
    target = KnockoutTarget.ENTIRE_SUBJ
    target = KnockoutTarget.SUBJ_FIRST
    target = KnockoutTarget.AFTER_SUBJ

    for factor in [root_factor ** (i + 1) for i in range(6)]:
        for category in layer_classification:
            evaluator = IncreaseDeltaEvaluator(model, tokenizer, device, target, layer_classification[category], factor, args.show_eval_progress)

            _, acc = evaluator.knockout_eval(knowns_df, layers_of_interest, KnockoutMode.INCREASE_DELTA)
            
            performance['layers'].append(str(layers_of_interest))
            performance['factor'].append(factor)
            performance['category'].append(category)
            performance['acc'].append(acc)

            # save to csv
            df = pd.DataFrame(performance)
            print(df)
            out_fname = args.output_dir / f"{args.interfere_mode}_{args.model_size}_target_{target}_layer_neighborhood_{max(layers_of_interest)}-{min(layers_of_interest)}_root_decay_factor_{root_factor}.csv"
            if out_fname.exists():
                os.remove(out_fname)
            df.to_csv(out_fname)
    
    df = pd.DataFrame(performance)
    out_fname = args.output_dir / f"{args.interfere_mode}_{args.model_size}_target_{target}_layer_neighborhood_{max(layers_of_interest)}-{min(layers_of_interest)}_root_decay_factor_{root_factor}.csv"
    if out_fname.exists():
        os.remove(out_fname)
    df.to_csv(out_fname)


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, default='2.8B', help="Model size")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    model, tokenizer, device = setup_model(args.model_size)
    knowns_df = load_knowns()
    

    # If we do attention knockout:
    layers_of_interest = sorted([63, 62, 61, 60, 59, 58, 57, 56])
    layer_classification = DecayNormClassifier(norm=1).classify_model(model.backbone)
    factor = {'max': 1, 'min': 0.9, 'mid': 1.1}
    mask = {'max':0, 'min': 1, 'mid': 1}
    evaluator = AdaptationEvaluator(model, tokenizer, device, layer_classification, factor, mask, True)
    
    _, acc = evaluator.knockout_eval(knowns_df, layers_of_interest, KnockoutMode.INCREASE_DELTA)
    print(acc)
    

if __name__ == "__main__":
    main()
