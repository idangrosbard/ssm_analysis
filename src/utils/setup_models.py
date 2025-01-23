import os
from typing import TYPE_CHECKING, Optional, Tuple, Union, assert_never

import torch
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    MambaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

import src.models.minimal_mamba2 as minimal_mamba2
import src.models.minimal_mamba2_new as minimal_mamba2_new
from src.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID
from src.models.minimal_mamba1 import Mamba
from src.types import MODEL_ARCH

if TYPE_CHECKING:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def setup_mamba_model(
    model_size: str = "2.8B",
) -> Tuple[MambaForCausalLM, PreTrainedTokenizer | PreTrainedTokenizerFast, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = get_tokenizer_and_model(MODEL_ARCH.MAMBA1, model_size)
    assert isinstance(model, MambaForCausalLM)
    return model, tokenizer, device


def _get_tokenizer_id(model_arch: MODEL_ARCH, model_id: str) -> str:
    match model_arch:
        case (
            MODEL_ARCH.MAMBA1
            | MODEL_ARCH.MAMBA2
            | MODEL_ARCH.MINIMAL_MAMBA1
            | MODEL_ARCH.MINIMAL_MAMBA2
            | MODEL_ARCH.MINIMAL_MAMBA2_new
        ):
            return "EleutherAI/gpt-neox-20b"
        case MODEL_ARCH.LLAMA2 | MODEL_ARCH.LLAMA3_2:
            return model_id
        case _:
            assert_never(model_arch)


def get_tokenizer_and_model(
    model_arch: MODEL_ARCH, model_size: str, device: Optional[torch.device] = None
) -> tuple[
    PreTrainedTokenizer | PreTrainedTokenizerFast,
    Union[
        Mamba,
        minimal_mamba2.Mamba2LMHeadModel,
        minimal_mamba2_new.Mamba2LMHeadModel,
        PreTrainedModel,
        MambaForCausalLM,
        Mamba,
        LlamaForCausalLM,
        "MambaLMHeadModel",
    ],
]:
    if os.getenv("HUGGINGFACE_TOKEN") is not None:
        login(token=os.getenv("HUGGINGFACE_TOKEN"))

    minimal_kwargs = {
        "device": device,
        "device_map": "auto" if device is None else None,
    }

    model_id = MODEL_SIZES_PER_ARCH_TO_MODEL_ID[model_arch][model_size]
    tokenizer = AutoTokenizer.from_pretrained(_get_tokenizer_id(model_arch, model_id))
    tokenizer.pad_token = tokenizer.eos_token

    match model_arch:
        case MODEL_ARCH.MINIMAL_MAMBA1:
            model = Mamba.from_pretrained(model_id, **minimal_kwargs)  # type: ignore
        case MODEL_ARCH.MINIMAL_MAMBA2:
            model = minimal_mamba2.Mamba2LMHeadModel.from_pretrained(model_id, **minimal_kwargs)  # type: ignore
        case MODEL_ARCH.MINIMAL_MAMBA2_new:
            model = minimal_mamba2_new.Mamba2LMHeadModel.from_pretrained(model_id, **minimal_kwargs)  # type: ignore
        case MODEL_ARCH.MAMBA1:
            if device:
                model = MambaForCausalLM.from_pretrained(model_id)
                model.to(device)  # type: ignore
            else:
                model = MambaForCausalLM.from_pretrained(model_id, device_map="auto")
        case MODEL_ARCH.MAMBA2:
            if not device:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

            model = MambaLMHeadModel.from_pretrained(model_id, device=device)
        case MODEL_ARCH.LLAMA2 | MODEL_ARCH.LLAMA3_2:
            if device:
                model = LlamaForCausalLM.from_pretrained(model_id)
                model.to(device)  # type: ignore
            else:
                model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto")
        case _:
            assert_never(model_arch)

    return tokenizer, model
