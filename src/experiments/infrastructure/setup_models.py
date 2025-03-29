import os
from typing import Optional, assert_never

from huggingface_hub import login

from src.core.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID, is_falcon
from src.core.types import MODEL_ARCH, TDevice, TModel, TModelID, TModelSize, TTokenizer


def _get_tokenizer_id(model_id: str) -> str:
    if model_id.startswith("state-spaces/"):
        return "EleutherAI/gpt-neox-20b"
    else:
        return model_id


MODEL_TOKENIZER_CACHE: dict[TModelID, TTokenizer] = {}


def get_tokenizer(model_arch: MODEL_ARCH, model_size: TModelSize) -> TTokenizer:
    from transformers import AutoTokenizer

    model_id = MODEL_SIZES_PER_ARCH_TO_MODEL_ID[model_arch][model_size]
    if model_id in MODEL_TOKENIZER_CACHE:
        return MODEL_TOKENIZER_CACHE[model_id]
    tokenizer = AutoTokenizer.from_pretrained(_get_tokenizer_id(model_id))
    tokenizer.pad_token = tokenizer.eos_token
    MODEL_TOKENIZER_CACHE[model_id] = tokenizer
    return tokenizer


def get_tokenizer_and_model(
    model_arch: MODEL_ARCH, model_size: TModelSize, device: Optional[TDevice] = None
) -> tuple[TTokenizer, TModel]:
    if os.getenv("HUGGINGFACE_TOKEN") is not None:
        login(token=os.getenv("HUGGINGFACE_TOKEN"))

    minimal_kwargs = {
        "device": device,
        "device_map": "auto" if device is None else None,
    }

    model_id = MODEL_SIZES_PER_ARCH_TO_MODEL_ID[model_arch][model_size]
    tokenizer = get_tokenizer(model_arch, model_size)

    match model_arch:
        case MODEL_ARCH.MAMBA2:
            import src.experiments.knockout.mamba.mamba2.minimal_mamba2 as minimal_mamba2

            model = minimal_mamba2.Mamba2LMHeadModel.from_pretrained(model_id, **minimal_kwargs)  # type: ignore
        case MODEL_ARCH.MAMBA1:
            if is_falcon(model_size):
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
            else:
                from transformers import MambaForCausalLM

                if device:
                    model = MambaForCausalLM.from_pretrained(model_id)
                    model.to(device)  # type: ignore
                else:
                    model = MambaForCausalLM.from_pretrained(model_id, device_map="auto")
        case MODEL_ARCH.LLAMA2 | MODEL_ARCH.LLAMA3_2 | MODEL_ARCH.LLAMA3:
            from transformers import LlamaForCausalLM

            if device:
                model = LlamaForCausalLM.from_pretrained(model_id)
                model.to(device)  # type: ignore
            else:
                model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto")
        case MODEL_ARCH.GPT2:
            from transformers import GPT2LMHeadModel

            model = GPT2LMHeadModel.from_pretrained(model_id, device_map="auto")
        case _:
            assert_never(model_arch)

    return tokenizer, model
