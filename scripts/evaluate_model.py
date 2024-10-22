from typing import NamedTuple
from typing import Optional
from typing import assert_never

import wget
import pandas as pd
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from scripts.create_slurm_file import run_slurm
from src.models.minimal_mamba1 import Mamba
from src.models.minimal_mamba2 import Mamba2LMHeadModel


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", type=str, choices={'mamba', 'minimal_mamba1', 'minimal_mamba2'},
                        default="mamba")
    parser.add_argument("--model_size", type=str, default="130M")  # 130M, 2.8B, 2.7B
    parser.add_argument("--drop_subject", action='store_true')
    parser.add_argument("--output_file", type=Path, default=None)
    parser.add_argument("--drop_subj_last_token", action='store_true')
    parser.add_argument("--with_3_dots", type=int, default=0)
    parser.add_argument("--with_slurm", action='store_true', help="Run the script with slurm")
    return parser.parse_args()


def get_subj_idx(input: str, subj: str, tokenizer: AutoTokenizer, last: bool = True):
    prefix = input.split(subj)[0]
    sent2subj = prefix
    if last:
        sent2subj = prefix + subj

    sent2subj_tokens = tokenizer(sent2subj)["input_ids"]
    return len(sent2subj_tokens) - 1


def get_tokenizer_and_model(model_arch: str, model_size: str, device) -> (
        AutoTokenizer, MambaForCausalLM | Mamba | Mamba2LMHeadModel):
    if model_arch == "mamba":
        tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
        model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    elif model_arch == "minimal_mamba1":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = Mamba.from_pretrained(f"state-spaces/mamba-{model_size}")
    elif model_arch == "minimal_mamba2":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = Mamba2LMHeadModel.from_pretrained(f"state-spaces/mamba2-{model_size}", device=device)
    else:
        assert_never(model_arch)

    model.to(device)
    return tokenizer, model


def main(
        model_arch: str,
        model_size: str = "2.8B",
        drop_subject: bool = False,
        drop_subj_last_token: bool = False,
        with_3_dots: bool = False,
        output_file: Optional[Path] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path("known_1000.json").exists():
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json").set_index('known_id')
    print(knowns_df)

    tokenizer, model = get_tokenizer_and_model(model_arch, model_size, device)
    model.eval()

    acc = 0

    knowns_df['model_correct'] = False

    with torch.no_grad():
        pbar = tqdm(knowns_df.index, total=len(knowns_df))
        for idx in pbar:
            input = knowns_df.loc[idx, "prompt"]
            target = knowns_df.loc[idx, "attribute"]

            if with_3_dots:
                input += " ..."
            if drop_subject:
                input = input.replace(knowns_df.loc[idx, "subject"], '')
            elif drop_subj_last_token:
                subj_idx = get_subj_idx(input, knowns_df.loc[idx, "subject"], tokenizer)

            input_ids = tokenizer(input)["input_ids"]

            if drop_subj_last_token:
                input_ids = input_ids[:subj_idx] + input_ids[subj_idx + 1:]

            input_ids = torch.Tensor([input_ids]).long().to(device)

            with torch.no_grad():
                if model_arch == "minimal_mamba1":
                    out = model(input_ids)
                    logits = out
                elif model_arch == "minimal_mamba2":
                    out = model(input_ids[0], with_chunk_handling=True)
                    logits = out[0]
                elif model_arch == "mamba":
                    out = model(input_ids)
                    logits = out.logits
                else:
                    assert_never(model_arch)

            # print the id of subj_tokens

            decoded = tokenizer.decode(logits.argmax(dim=-1).squeeze())
            last_word = decoded.split(' ')[-1]

            knowns_df.loc[idx, 'model_correct'] = last_word == target[:len(last_word)]
            knowns_df.loc[idx, 'model_output'] = last_word

            acc += float(last_word == target[:len(last_word)]) / len(knowns_df)

    print(acc)
    if not output_file:
        output_file = Path(f'known_1000_{model_arch}_{model_size}_correct.csv')
    knowns_df.to_csv(output_file)


def run_with_slurm(
        model_arch: str,
        model_size: str,
        drop_subject: bool,
        drop_subj_last_token: bool,
        with_3_dots: bool,
):
    project_dir = Path(__file__).parent.parent
    experiment_name = f"{model_arch}_{model_size}"
    if with_3_dots:
        experiment_name += "_with_3_dots"
    output_dir = project_dir / "output" / experiment_name / "evaluate"
    script_path = Path(__file__).resolve()

    store_args = {}
    if drop_subject:
        store_args["drop_subject"] = True
    if drop_subj_last_token:
        store_args["drop_subj_last_token"] = True

    run_slurm(
        experiment_name=experiment_name,
        output_dir=output_dir,
        script_path=script_path,
        script_args=dict(
            model_arch=model_arch,
            model_size=model_size,
            with_3_dots=with_3_dots,
            **store_args,
            output_file=output_dir / f'known_1000_correct.csv'
        )
    )

if __name__ == "__main__":
    args = get_args()
    assert not (args.drop_subject and args.drop_subj_last_token)
    args = dict(args._get_kwargs())
    with_slurm = args.pop("with_slurm")
    if with_slurm:
        del args["output_file"]
        run_with_slurm(**args)
    else:
        main(**args)

# python /home/yandex/DL20232024a/nirendy/repos/ssm_analysis/scripts/evaluate_model.py --model_arch mamba --model_size 2.8B --with_slurm
# python /home/yandex/DL20232024a/nirendy/repos/ssm_analysis/scripts/evaluate_model.py --model_arch minimal_mamba2 --model_size 2.8B --with_slurm
# python /home/yandex/DL20232024a/nirendy/repos/ssm_analysis/scripts/evaluate_model.py --model_arch minimal_mamba1 --model_size 2.8B --with_slurm
