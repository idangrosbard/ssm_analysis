import torch
from transformers import AutoTokenizer
from torch import nn
from typing import Tuple
import pandas as pd
from tqdm import tqdm


def evaluate_sample(model: nn.Module, tokenizer: AutoTokenizer, x: str, y: str, device: torch.device) -> bool:
    x_ids = tokenizer(x)["input_ids"]

    x_ids = torch.Tensor([x_ids]).long().to(device)
    
    out = model(x_ids)
    decoded = tokenizer.decode(out.logits.argmax(dim=-1).squeeze())
    last_word = decoded.split(' ')[-1]

    return last_word == y[:len(last_word)]


def evaluate_model(model: nn.Module, tokenizer: AutoTokenizer, knowns_df: pd.DataFrame, device: torch.device) -> Tuple[pd.DataFrame, float]:
    
    model.to(device)
    model.eval()

    acc = 0
    knowns_df['model_correct'] = False

    with torch.no_grad():
        pbar = tqdm(knowns_df.index, total=len(knowns_df))
        for idx in pbar:
            input = knowns_df.loc[idx, "prompt"]
            target = knowns_df.loc[idx, "attribute"]

            correct = evaluate_sample(model, tokenizer, input, target, device)

            knowns_df.loc[idx, 'model_correct'] = correct

            acc += float(correct) / len(knowns_df)
    
    return knowns_df, acc