from collections import defaultdict
from dataclasses import dataclass
from typing import assert_never

import torch

from src.types import TNum2Mask, TokenType, TPromptData, TTokenizer


def get_last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, :]


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def get_top_k_outputs_and_probs(logits: torch.Tensor, tokenizer, top_k: int):
    next_probs = logits_to_probs(get_last_token_logits(logits))
    top_probs, top_indices = torch.topk(next_probs, top_k)
    top_outputs = [
        (idx.item(), str(tokenizer.decode([idx])), prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])
    ]
    return top_outputs


def get_top_outputs(probs, tokenizer, top_k):
    # Get the top 5 outputs and their probs
    top_probs, top_indices = map(torch.Tensor.tolist, torch.topk(torch.Tensor(probs), top_k))
    top_tokens = list(map(tokenizer.batch_decode, top_indices))
    return list(
        map(
            list,
            map(
                lambda x: zip(*x),
                zip(
                    top_indices,
                    top_tokens,
                    top_probs,
                ),
            ),
        )
    )


# Taken from https://github.com/google-research/google-research/blob/master/dissecting_factual_predictions/utils.py
def decode_tokens(tokenizer: TTokenizer, token_array: torch.Tensor):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(
    tokenizer,
    token_array,
    substring,
) -> tuple[int, int]:
    """Find the tokens corresponding to the given substring in token_array."""
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)  # type: ignore
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    assert tok_start is not None and tok_end is not None, "Token range not found"
    return (tok_start, tok_end)


@dataclass
class Prompt:
    prompt_row: TPromptData

    @property
    def prompt(self):
        return self.prompt_row["prompt"]

    @property
    def subject(self):
        return self.prompt_row["subject"]

    @property
    def true_word(self):
        return self.prompt_row["target_true"]

    @property
    def base_prob(self):
        return self.prompt_row["true_prob"]

    def true_id(self, tokenizer, device) -> torch.Tensor:
        return tokenizer(self.true_word, return_tensors="pt", padding=True).input_ids.to(device="cpu")

    def input_ids(self, tokenizer, device) -> torch.Tensor:
        return tokenizer(self.prompt, return_tensors="pt", padding=True).input_ids.to(device=device)


def get_num_to_masks(
    prompt: Prompt,
    tokenizer,
    window: list[int],
    knockout_src: TokenType,
    knockout_target: TokenType,
    device,
) -> tuple[TNum2Mask, bool]:
    input_ids = prompt.input_ids(tokenizer, device)
    num_to_masks = TNum2Mask(defaultdict(list))
    first_token = False

    last_idx = input_ids.shape[1] - 1
    tok_start, tok_end = find_token_range(tokenizer, input_ids[0], prompt.subject)
    subject_tokens = list(range(tok_start, tok_end))
    if 0 in subject_tokens:
        first_token = True

    def get_knockout_idx(knockout: TokenType):
        if knockout == TokenType.first:
            return [0]
        elif knockout == TokenType.last:
            return [last_idx]
        elif knockout == TokenType.subject:
            return subject_tokens
        elif knockout == TokenType.relation:
            return [i for i in range(last_idx + 1) if i not in subject_tokens]
        elif knockout == TokenType.context:
            return [i for i in range(subject_tokens[0])]
        else:
            assert_never(knockout_src)

    src_idx = get_knockout_idx(knockout_src)
    target_idx = get_knockout_idx(knockout_target)

    for layer in window:
        for src in src_idx:
            for target in target_idx:
                num_to_masks[layer].append((target, src))

    return num_to_masks, first_token


def get_prompt_row(data: TPromptData, prompt_idx: int) -> Prompt:
    return Prompt(prompt_row=data.loc[prompt_idx])  # type: ignore
