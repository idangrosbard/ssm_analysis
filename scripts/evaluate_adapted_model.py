import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from src.datasets.download_dataset import load_knowns_pd
from src.knockout.adapted_model import AdaptationEvaluator
from src.knockout.ssm_knockout.ssm_classifier import SSMClassifier, DecayNormClassifier, SSMClassifierStub
from argparse import ArgumentParser, Namespace
from src.utils.setup_models import setup_mamba_model
from src.knockout.knockout_mode import KnockoutMode

from src.datasets.download_dataset import load_dataset
from src.types import DatasetArgs
from src.types import DATASETS



def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, default='2.8B', help="Model size")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    model, tokenizer, device = setup_mamba_model(args.model_size)
    knowns_df = pd.DataFrame(load_dataset(DatasetArgs(name=DATASETS.KNOWN_1000, splits=['test'])))
    knowns_df['attribute'] = knowns_df['attribute'].apply(lambda x: x[1:])

    # If we do attention knockout:
    layers_of_interest = sorted([63, 62, 61, 60, 59, 58, 57, 56])
    layer_classification = DecayNormClassifier(norm=1).classify_model(model.backbone)
    factor = {'max': 1, 'min': 0.5, 'mid': 1}
    mask = {'max':0, 'min': 1, 'mid': 1}
    evaluator = AdaptationEvaluator(model, tokenizer, device, layer_classification, factor, mask, True)
    
    _, acc = evaluator.knockout_eval(knowns_df, layers_of_interest, KnockoutMode.INCREASE_DELTA)
    print(acc)
    

if __name__ == "__main__":
    main()
