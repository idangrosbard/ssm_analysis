import torch
from transformers import AutoTokenizer
from torch import nn
from typing import Tuple
import pandas as pd
from tqdm import tqdm


def evaluate_sample(model: nn.Module, tokenizer: AutoTokenizer, x: str, y: str, device: torch.device) -> float:
    x_ids = tokenizer(x)["input_ids"]

    x_ids = torch.Tensor([x_ids]).long().to(device)
    
    out = model(x_ids)

    last_token_logits = out.logits[0, -1, :]
    last_token_probs = torch.softmax(last_token_logits, dim=-1)
    last_token_id = tokenizer(y)["input_ids"][0]
    correct_prob = last_token_probs[last_token_id]

    return correct_prob.item()


def evaluate_model(model: nn.Module, tokenizer: AutoTokenizer, knowns_df: pd.DataFrame, device: torch.device) -> Tuple[pd.DataFrame, float]:
    
    model.to(device)
    model.eval()

    mean_prob_diff = 0
    knowns_df['knockout_prob'] = False

    with torch.no_grad():
        pbar = tqdm(knowns_df.index, total=len(knowns_df), disable=True)
        for idx in pbar:
            input = knowns_df.loc[idx, "prompt"]
            target = knowns_df.loc[idx, "attribute"]
            base_prob = knowns_df.loc[idx, "true_prob"]
            
            prob = evaluate_sample(model, tokenizer, input, target, device)

            knowns_df.loc[idx, "knockout_prob"] = prob

            mean_prob_diff += (prob - base_prob) / len(knowns_df)
    
    return knowns_df, mean_prob_diff