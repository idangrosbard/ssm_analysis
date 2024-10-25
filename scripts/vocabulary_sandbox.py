from transformers import AutoTokenizer, MambaForCausalLM
from typing import Callable
import pandas as pd
import torch


class DecoderHook(Callable):
    def __init__(self, input: str, tokenizer: AutoTokenizer, E: torch.Tensor, tokens: torch.Tensor, k: int = 5, layer: int | str | torch.nn.Module = None):
        self.tokenizer = tokenizer
        self.k = k
        self.E = E
        self.layer = layer
        self.tokens = tokens
        self.input = input
        self.output_df = {'t': [], 'token': [], 'decoded': [], 'score': [], 'rank': []}

    def hook(self, module, inp, out):
        embeddings = out
        self.decode(embeddings)

    def decode(self, embeddings):
        T = embeddings.shape[1]
        scores = embeddings @ self.E.T
        topk = torch.topk(scores, k, dim=-1)
        topk_values = topk.values
        topk_indices = topk.indices

        for t in range(T):
            print()
            print(self.layer)
            token = self.tokenizer.convert_ids_to_tokens(self.tokens[0, t].item())
            # print(f"Token: {token}")
            # print(f"Top {k} tokens:")
            for i in range(k):
                decoded = self.tokenizer.convert_ids_to_tokens(topk_indices[0, t, i].item())
                score = topk_values[0, t, i].item()
                self.output_df['t'].append(t)
                self.output_df['token'].append(token)
                self.output_df['decoded'].append(decoded)
                self.output_df['score'].append(score)
                self.output_df['rank'].append(i + 1)
                # print(f"  {decoded}: {score}")

    def __call__(self, module, inp, out):
        return self.hook(module, inp, out)



if __name__ == '__main__':
    model_size = '2.8B'
    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.eval()
    sentence = "The quick brown fox jumps over the lazy dog."
    sentence = "Microsoft"
    sentence = "Bill Gates"
    # sentence = "French"
    # sentence = "The"
    # sentence = "Cheetah"
    tokens = tokenizer(sentence, return_tensors='pt')["input_ids"]
    print(tokens)
    T = tokens.shape[1]
    embeddings = model.backbone.embeddings(tokens)
    E = model.backbone.embeddings.weight
    
    k = 20
    hooks = []
    handles = []
    
    for i, mod in enumerate(model.backbone.layers):
        hooks.append(DecoderHook(sentence, tokenizer, E, tokens, k, layer=i))
        handles.append(mod.mixer.register_forward_hook(hooks[-1]))

    out = model(tokens)

    for handle in handles:
        handle.remove()

    dfs = []
    for hook in hooks:
        dfs.append(pd.DataFrame(hook.output_df))
        dfs[-1]['layer'] = hook.layer
    
    df = pd.concat(dfs)
    df['input'] = sentence
    df.to_csv(f"decoded_inputs_{sentence}.csv")