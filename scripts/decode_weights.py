import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import plotly.express as px
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, MambaModel

from src.weight_analysis import (
    decode,
    get_singular_values,
    get_topk_singular_vectors,
    low_rank_approx_error_df,
    plot,
)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, choices={"130M", "2.8B"}, default="130m")
    parser.add_argument("--output_dir", type=Path, default=Path("resources"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--svd", action="store_true")
    parser.add_argument("--k_components", type=int, default=24)
    return parser.parse_args()


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{args.model_size}-hf")
    model = MambaModel.from_pretrained(f"state-spaces/mamba-{args.model_size}-hf")

    model.eval()
    dfs = []
    E = model.embeddings.weight
    for i, layer in tqdm(enumerate(model.layers), total=len(model.layers)):
        w = layer.mixer.out_proj.weight.T
        if args.svd:
            w = get_topk_singular_vectors(w, args.k_components, False)

        df = decode(w, E, args.k, tokenizer, None)
        df["layer"] = i
        dfs.append(df)

    dfs = pd.concat(dfs)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_fname = args.output_dir / f"decoded_{args.model_size}{'svd' if args.svd else None}.csv"

    dfs.to_csv(out_fname)


def plot_singular_values():
    args = get_args()
    model = MambaModel.from_pretrained(f"state-spaces/mamba-{args.model_size}-hf")
    model.eval()
    singular_values = []
    layer_indices = []
    value_rankings = []
    for i, layer in tqdm(enumerate(model.layers), total=len(model.layers)):
        w = layer.mixer.out_proj.weight.detach().cpu()
        s = get_singular_values(w)
        singular_values.append(s)
        layer_indices.append(torch.ones_like(s) * i)
        value_rankings.append(torch.arange(len(s)))

    singular_values = torch.cat(singular_values)
    layer_indices = torch.cat(layer_indices)
    value_rankings = torch.cat(value_rankings)
    df = pd.DataFrame(
        {
            "layer": layer_indices.numpy(),
            "singular_value": singular_values.numpy(),
            "rank": value_rankings.numpy(),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fname = args.output_dir / f"singular_values_{args.model_size}.html"
    plot(df).write_html(fname)


def plot_low_rank_approx_err():
    args = get_args()
    model = MambaModel.from_pretrained(f"state-spaces/mamba-{args.model_size}-hf")
    model.eval()
    dfs = []

    for i, layer in tqdm(enumerate(model.layers), total=len(model.layers)):
        w = layer.mixer.out_proj.weight.detach().cpu()
        df = low_rank_approx_error_df(get_singular_values(w))
        df["layer"] = i
        dfs.append(df)

    dfs = pd.concat(dfs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    px.line(dfs, x="rank", y="spectral_error", color="layer").write_html(
        args.output_dir / f"low_rank_approx_{args.model_size}.html"
    )


if __name__ == "__main__":
    plot_low_rank_approx_err()
    plot_singular_values()
    main()
