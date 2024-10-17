from transformers import AutoTokenizer
from typing import Tuple, Iterable
from .knockout_target import KnockoutTarget
import random


def get_subj_idx(input: str, subj: str, tokenizer: AutoTokenizer) -> Tuple[int,int]:
    prefix = input.split(subj)[0]
    sent2subj = prefix
    
    if prefix == "":
        sent2subj = subj
    else:
        sent2subj = prefix + ' ' + subj

    sent2subj_tokens = tokenizer(sent2subj)["input_ids"]
    prefix_tokens = tokenizer(prefix)["input_ids"]
    return (len(prefix_tokens), len(sent2subj_tokens))


def check_intersect(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    assert a[0] <= a[1]
    assert b[0] <= b[1]

    if a[0] < b[0] < a[1]:
        return True
    if a[0] < b[1] < a[1]:
        return True
    if b[0] < a[0] < b[1]:
        return True
    if b[0] < a[1] < b[1]:
        return True
    return False


def random_non_subj(input: str, subj: str, tokenizer: AutoTokenizer, single_token: bool):
    subj_idx = get_subj_idx(input, subj, tokenizer)
    while True:
        if single_token:
            # if we want a single token, we can just pick a random token
            start = random.randint(0, len(tokenizer(input)["input_ids"]))
            target_idx = (start, start)
        else:
            # otherwise, we need to pick a span
            a = random.randint(0, len(tokenizer(input)["input_ids"]))
            b = random.randint(0, len(tokenizer(input)["input_ids"]))
            start, end = min(a, b), max(a, b)

            # end is non-inclusive, so if they're the same we need to increment `end`
            target_idx = (start, end)
        
        # If the two don't intersect, we're good
        if not check_intersect(subj_idx, target_idx):
            break
    
    return target_idx


def choose_knockout_target(input: str, subj: str, tokenizer: AutoTokenizer, target: KnockoutTarget) -> Iterable[int]:
    if target == KnockoutTarget.ENTIRE_SUBJ:
        first, last = get_subj_idx(input, subj, tokenizer)
        last = last - 1
    elif target == KnockoutTarget.SUBJ_LAST:
        first, last = get_subj_idx(input, subj, tokenizer)
        last = last - 1
        first = last
    elif target == KnockoutTarget.FIRST:
        first = 0
        last = first
    elif target == KnockoutTarget.LAST:
        first = len(tokenizer(input)["input_ids"]) - 1
        last = first
    elif target == KnockoutTarget.RANDOM:
        first, last = random_non_subj(input, subj, tokenizer, single_token=True)
    elif target == KnockoutTarget.RANDOM_SPAN:
        first, last = random_non_subj(input, subj, tokenizer, single_token=False)
    elif target == KnockoutTarget.ALL_CONTEXT:
        first = 0
        last = len(tokenizer(input)["input_ids"]) - 1
    elif target == KnockoutTarget.SUBJ_CONTEXT:
        first, last = get_subj_idx(input, subj, tokenizer)
        last = first - 1
        first = 0
    return {i for i in range(first, last + 1)}


def is_last_token_subj(input: str, subj: str, tokenizer: AutoTokenizer) -> Tuple[int,int]:
    last_token_idx = len(tokenizer(input)["input_ids"])
    last_subj_idx = get_subj_idx(input, subj, tokenizer)[1]
    return (last_token_idx == last_subj_idx)
