import wget
import pandas as pd
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, choices={'130M', '2.8B'}, default="130M")
    parser.add_argument("--drop_subject", action='store_true')
    parser.add_argument("--drop_subject_last_token", action='store_true')
    
    return parser.parse_args()


def get_subj_idx(input: str, subj: str, tokenizer: AutoTokenizer, last: bool = True):
    prefix = input.split(subj)[0]
    sent2subj = prefix
    if last:
        sent2subj = prefix + subj
        
    sent2subj_tokens = tokenizer(sent2subj)["input_ids"]
    return len(sent2subj_tokens) - 1


def main(model_size: str = "2.8B", drop_subject: bool = False, drop_subj_last_token: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path("known_1000.json").exists():
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json").set_index('known_id')
    print(knowns_df)

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.to(device)

    model.eval()

    acc = 0

    knowns_df['model_correct'] = False

    with torch.no_grad():
        pbar = tqdm(knowns_df.index, total=len(knowns_df))
        for idx in pbar:
            input = knowns_df.loc[idx, "prompt"]
            target = knowns_df.loc[idx, "attribute"]

            if drop_subject:
                input = input.replace(knowns_df.loc[idx, "subject"], '')
            elif drop_subj_last_token:
                subj_idx = get_subj_idx(input, knowns_df.loc[idx, "subject"], tokenizer)

            input_ids = tokenizer(input)["input_ids"]

            if drop_subj_last_token:
                input_ids = input_ids[:subj_idx] + input_ids[subj_idx+1:]

            input_ids = torch.Tensor([input_ids]).long().to(device)

            # print the id of subj_tokens
            
            out = model(input_ids)
            decoded = tokenizer.decode(out.logits.argmax(dim=-1).squeeze())
            last_word = decoded.split(' ')[-1]

            knowns_df.loc[idx, 'model_correct'] = last_word == target[:len(last_word)]

            acc += float(last_word == target[:len(last_word)]) / len(knowns_df)
    
    print(acc)
    knowns_df.to_csv(f'known_1000_{model_size}_correct.csv')

if __name__ == "__main__":
    args = get_args()
    assert not (args.drop_subject and args.drop_subject_last_token)
    main(args.model_size, args.drop_subject)