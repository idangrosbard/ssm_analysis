import wget
import pandas as pd
from transformers import AutoTokenizer, MambaForCausalLM
import torch
from tqdm import tqdm
from pathlib import Path



def main(model_size: str = "2.8B"):
    if not Path("known_1000.json").exists():
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json").set_index('known_id')
    print(knowns_df)

    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.cuda()

    model.eval()

    acc = 0

    knowns_df['model_correct'] = False

    with torch.no_grad():
        pbar = tqdm(knowns_df.index, total=len(knowns_df))
        for idx in pbar:
            input = knowns_df.loc[idx, "prompt"]
            target = knowns_df.loc[idx, "attribute"]

            input_ids = tokenizer(input, return_tensors="pt")["input_ids"].cuda()
            # print the id of subj_tokens
            
            out = model(input_ids)
            decoded = tokenizer.decode(out.logits.argmax(dim=-1).squeeze())
            last_word = decoded.split(' ')[-1]

            knowns_df.loc[idx, 'model_correct'] = last_word == target[:len(last_word)]

            acc += float(last_word == target[:len(last_word)]) / len(knowns_df)
    
    print(acc)
    knowns_df.to_csv(f'known_1000_{model_size}_correct.csv')

if __name__ == "__main__":
    main()