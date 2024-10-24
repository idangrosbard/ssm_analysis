from typing import Optional
from typing import Tuple

import torch
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import MambaForCausalLM
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast

from src.models.minimal_mamba1 import Mamba
from src.models.minimal_mamba2 import Mamba2LMHeadModel


def setup_model(model_size: str = "2.8B") -> Tuple[MambaForCausalLM, AutoTokenizer, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = get_tokenizer_and_model("minimal_mamba1", model_size, device)
    return model, tokenizer, device


def get_tokenizer_and_model(model_arch: str, model_size: str, device: Optional[torch.device] = None) -> tuple[
    PreTrainedTokenizer | PreTrainedTokenizerFast,
    PreTrainedModel | MambaForCausalLM | Mamba | Mamba2LMHeadModel | LlamaForCausalLM,
]:
    if model_arch == "mamba":
        tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
        model = MambaForCausalLM.from_pretrained(
            f"state-spaces/mamba-{model_size}-hf"
        )  # 130M, 2.8B
    elif model_arch == "minimal_mamba1":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = Mamba.from_pretrained(f"state-spaces/mamba-{model_size}")
    elif model_arch == "minimal_mamba2":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = Mamba2LMHeadModel.from_pretrained(
            f"state-spaces/mamba2-{model_size}", device=device  # 130M, 2.7B
        )
    elif model_arch == "llama2":
        model_name = f"meta-llama/Llama2-{model_size}-hf"  # 7b
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama2-{model_size}-hf")
        model = LlamaForCausalLM.from_pretrained(
            f"meta-llama/Llama-{model_size}-hf"
        )  # 7
    elif model_arch == "llama3.2":
        model_name = f"meta-llama/Llama-3.2-{model_size}"  # 1B, 3B
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name)
    else:
        assert False, f"model_arch {model_arch} not supported"

    if device:
        model.to(device)

    return tokenizer, model
