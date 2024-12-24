import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.datasets.download_dataset import load_knowns_pd
from src.evaluate import evaluate_model
from src.weight_analysis import get_low_rank_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, choices={"130M", "2.8B"}, default="130M")
    parser.add_argument("--rank", type=int, default=768)
    parser.add_argument("--output_dir", type=Path, default=Path("resources"))
    parser.add_argument("--use_min_vals", action="store_true")

    return parser.parse_args()


def main(
    output_dir: Path,
    model_size: str = "2.8B",
    rank: int = 768,
    use_min_vals: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")

    knowns_df = load_knowns_pd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.eval()

    approx_backbone = get_low_rank_model(model.backbone, rank, not use_min_vals)
    approx_backbone.to(device)
    approx_backbone.eval()
    model.backbone = approx_backbone

    results, acc = evaluate_model(model, tokenizer, knowns_df, device)

    print(acc, rank)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / f"low_rank_eval_{model_size}_{rank}.csv")


if __name__ == "__main__":
    args = get_args()
    main(args.output_dir, args.model_size, args.rank, args.use_min_vals)
