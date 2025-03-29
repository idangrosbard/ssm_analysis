from typing import Iterable, Optional, Tuple

from torch import LongTensor, Tensor, nn
from transformers.models.llama.modeling_llama import (
    Cache,
    FlashAttentionKwargs,
    LlamaAttention,
    Unpack,
)

from src.experiments.knockout.llama.llama_attention_forward import llama_attention_forward


class LlamaAttentionKnockout(nn.Module):
    def __init__(self, inner: LlamaAttention, knockout_mask: Optional[Iterable[tuple[int, int]]] = None):
        super().__init__()
        self.inner = inner
        self.knockout_mask = knockout_mask

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[Tensor, Optional[Tensor]]:  # , Optional[Tuple[torch.Tensor]]]:
        # Call the original forward method
        return llama_attention_forward(
            self.inner,
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            knockout_mask=self.knockout_mask,
            **kwargs,
        )
