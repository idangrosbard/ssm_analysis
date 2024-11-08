import torch


def get_last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, :]


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def get_top_k_outputs_and_probs(
    logits: torch.Tensor, tokenizer, top_k: int
) -> list[tuple[int, str, float]]:
    next_probs = logits_to_probs(get_last_token_logits(logits))
    top_probs, top_indices = torch.topk(next_probs, top_k)
    top_outputs = [
        (idx.item(), str(tokenizer.decode([idx])), prob.item())
        for idx, prob in zip(top_indices[0], top_probs[0])
    ]
    return top_outputs
