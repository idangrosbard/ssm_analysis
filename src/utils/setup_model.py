from typing import Tuple
from transformers import AutoTokenizer, MambaForCausalLM
import torch


def setup_model(model_size: str = "2.8B") -> Tuple[MambaForCausalLM, AutoTokenizer, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return (model, tokenizer, device)