from typing import Optional, assert_never
from typing import Tuple

import torch
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import MambaForCausalLM
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from src.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID
from src.models.minimal_mamba1 import Mamba
from src.models.minimal_mamba2 import Mamba2LMHeadModel
from src.types import MODEL_ARCH

from huggingface_hub import login
import os

def setup_mamba_model(
    model_size: str = "2.8B",
) -> Tuple[
    MambaForCausalLM, PreTrainedTokenizer | PreTrainedTokenizerFast, torch.device
]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = get_tokenizer_and_model(
        MODEL_ARCH.MAMBA1, model_size, device=device
    )
    assert isinstance(model, MambaForCausalLM)
    return model, tokenizer, device


def _get_tokenizer_id(model_arch: MODEL_ARCH, model_id: str) -> str:
    match model_arch:
        case  MODEL_ARCH.MAMBA1 | MODEL_ARCH.MAMBA2 | MODEL_ARCH.MINIMAL_MAMBA1 | MODEL_ARCH.MINIMAL_MAMBA2:
            return f"EleutherAI/gpt-neox-20b"
        case MODEL_ARCH.LLAMA2 | MODEL_ARCH.LLAMA3_2:
            return model_id
        case _:
            assert_never(model_arch)


def get_tokenizer_and_model(
    model_arch: MODEL_ARCH, model_size: str, device: Optional[torch.device] = None
) -> tuple[
    PreTrainedTokenizer | PreTrainedTokenizerFast,
    PreTrainedModel | MambaForCausalLM | Mamba | Mamba2LMHeadModel | LlamaForCausalLM,
]:
    if os.getenv("HUGGINGFACE_TOKEN") is not None:
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
    
    minimal_kwargs = {
        "device": device,
        "device_map": "auto" if device is None else None,
    }

    model_id = MODEL_SIZES_PER_ARCH_TO_MODEL_ID[model_arch][model_size]
    tokenizer = AutoTokenizer.from_pretrained(_get_tokenizer_id(model_arch, model_id))

    match model_arch:
        case MODEL_ARCH.MINIMAL_MAMBA1:
            model = Mamba.from_pretrained(model_id, **minimal_kwargs)
        case MODEL_ARCH.MINIMAL_MAMBA2:
            model = Mamba2LMHeadModel.from_pretrained(model_id, **minimal_kwargs)
        case MODEL_ARCH.MAMBA1:
            if device:
                model = MambaForCausalLM.from_pretrained(model_id)
                model.to(device)
            else:
                model = MambaForCausalLM.from_pretrained(model_id, device_map="auto")
        case MODEL_ARCH.MAMBA2:
            if not device:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MambaLMHeadModel.from_pretrained(model_id, device=device)
        case MODEL_ARCH.LLAMA2 | MODEL_ARCH.LLAMA3_2:
            if device:
                model = LlamaForCausalLM.from_pretrained(model_id)
                model.to(device)
            else:
                model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto")
        case _:
            assert_never(model_arch)

    return tokenizer, model
