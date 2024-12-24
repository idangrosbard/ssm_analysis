from typing import Optional

import pandas as pd
import torch
from transformers import AutoTokenizer


def decode(
    embeddings: torch.Tensor,
    E: torch.Tensor,
    k: int,
    tokenizer: AutoTokenizer,
    tokens: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    output_df = {"t": [], "decoded": [], "score": [], "rank": []}
    if tokens is not None:
        output_df["token"] = []
        if len(tokens.shape) == 2:
            tokens = tokens.unsqueeze(dim=0)
    if len(embeddings.shape) == 3:
        embeddings = embeddings.squeeze(dim=0)
    T = embeddings.shape[0]
    scores = embeddings @ E.T
    topk = torch.topk(scores, k, dim=-1)
    topk_values = topk.values
    topk_indices = topk.indices

    for t in range(T):
        if tokens is not None:
            token = tokenizer.convert_ids_to_tokens(tokens[t].item())

        for i in range(k):
            if len(topk_indices.shape) == 3:
                topk_indices = topk_indices.squeeze(dim=0)
            decoded = tokenizer.convert_ids_to_tokens(topk_indices[t, i].item())
            score = topk_values[t, i].item()

            if tokens is not None:
                output_df["token"].append(token)

            output_df["t"].append(t)
            output_df["decoded"].append(decoded)
            output_df["score"].append(score)
            output_df["rank"].append(i + 1)

    return pd.DataFrame(output_df)
