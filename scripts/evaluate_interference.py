import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wget
import pandas as pd
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from tqdm import tqdm
from pathlib import Path
from src.hooks import SSMInterfereHook
from src.updates_ssm_ops import KnockoutMode
import plotly.express as px
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, choices={'130M', '2.8B'}, default="130M")
    parser.add_argument("--interfere_mode", type=str, choices={'ZERO_ATTENTION', 'ZERO_DELTA'}, default="ZERO_ATTENTION")
    return parser.parse_args()


def get_subj_idx(input: str, subj: str, tokenizer: AutoTokenizer, last: bool = True):
    prefix = input.split(subj)[0]
    sent2subj = prefix
    if last:
        sent2subj = prefix + subj
        
    sent2subj_tokens = tokenizer(sent2subj)["input_ids"]
    return len(sent2subj_tokens) - 1


def main(model_size: str = "2.8B", interefere_mode: KnockoutMode = KnockoutMode.ZERO_ATTENTION):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path("known_1000.json").exists():
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json").set_index('known_id')
    print(knowns_df)

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.to(device)

    model.eval()

    performance = {'acc': [], 'layer': []}

    knowns_df['model_correct'] = False



    with torch.no_grad():
        layers_pbar = tqdm(range(len(model.backbone.layers)))
        for i in layers_pbar:
            performance['acc'].append(0)
            performance['layer'].append(i)

            moi = model.backbone.layers[i].mixer

            hook = SSMInterfereHook(i, interefere_mode, -1)
            
            handle = moi.register_forward_hook(hook)

            pbar = tqdm(knowns_df.index, total=len(knowns_df))
            for idx in pbar:
                # Get relevant data
                input = knowns_df.loc[idx, "prompt"]
                target = knowns_df.loc[idx, "attribute"]
                subj = knowns_df.loc[idx, "subject"]

                # set subject token as knockout idx
                subj_token = get_subj_idx(input, subj, tokenizer)
                hook.knockout_idx = subj_token

                input_ids = tokenizer(input, return_tensors="pt")["input_ids"].to(device)
                out = model(input_ids)

                # get last decoded word
                decoded = tokenizer.decode(out.logits.argmax(dim=-1).squeeze())
                last_word = decoded.split(' ')[-1]

                # Update performance
                performance['acc'][-1] += float(last_word == target[:len(last_word)]) / len(knowns_df)

            print(performance['acc'][-1], i)
            handle.remove()
    
    df = pd.DataFrame(performance)
    px.line(data_frame=df, x='layer', y='acc', title='Accuracy per layer').write_html("ssm_interference_subject.html")



if __name__ == "__main__":
    args = get_args()
    main(args.model_size, KnockoutMode[args.interfere_mode])