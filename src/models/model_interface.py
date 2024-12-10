from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List, Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.types import MODEL_ARCH
from src.utils.setup_models import get_tokenizer_and_model

import src.models.minimal_mamba2_new as minimal_mamba2_new
from transformers import MambaForCausalLM


class ModelInterface(ABC):
    """Abstract interface for language models with attention knockout capability."""

    def __init__(
        self,
        model_arch: MODEL_ARCH,
        model_size: str,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    ):
        """Initialize the model with given size and device."""
        # x = get_tokenizer_and_model()
        self.tokenizer, self.model = get_tokenizer_and_model(
            model_arch, model_size, device
        )
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.device = self.model.device

    @abstractmethod
    def generate_logits(
        self,
        input_ids: torch.Tensor,
        attention: bool = False,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        """
        Generate logits for the input sequence with optional attention masking.

        Args:
            input_ids: Input token IDs
            attention: Whether to use attention mechanism
            num_to_masks: Dict mapping layer numbers to list of (idx1, idx2) tuples,
                        where idx1 won't get information from idx2

        Returns:
            Tuple of (next_token_logits, all_logits)
        """
        pass

    @abstractmethod
    def clean(self) -> None:
        """Clean up any resources or states if needed."""
        pass


class Mamba1Interface(ModelInterface):
    model: MambaForCausalLM

    def __init__(
        self,
        model_size: str,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    ):
        super().__init__(MODEL_ARCH.MAMBA1, model_size, device, tokenizer)

        self.handles = []

    def generate_logits(
        self,
        input_ids: Tensor,
        attention: bool = False,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        self.model.eval()
        hooks = []
        if num_to_masks is not None:
            # set up hooks
            for i in range(len(self.model.backbone.layers)):
                if i in num_to_masks:
                    # "mixer of interest" - moi
                    moi = self.model.backbone.layers[i].mixer

                    hooks.append(SSMInterfereHook(i, knockout_mode))

                    self.handles.append(moi.register_forward_hook(hooks[-1]))

            # set subject token as knockout idx
            knockout_indices = choose_knockout_target(
                input, subj, self.tokenizer, self.knockout_target
            )
            affected_target_indices = choose_knockout_target(
                input, subj, self.tokenizer, self.affected_target
            )

            for hook in hooks:
                hook.knockout_indices = knockout_indices
                hook.affected_outputs = affected_target_indices

        out = self.model(input_ids)

        return out.logits[-1]

    def clean(self) -> None:
        for handle in self.handles:
            handle.remove()


class Mamba2Interface(ModelInterface):
    model: minimal_mamba2_new.Mamba2LMHeadModel

    def __init__(
        self,
        model_size: str,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    ):
        super().__init__(MODEL_ARCH.MINIMAL_MAMBA2_new, model_size, device, tokenizer)

    def generate_logits(
        self,
        input_ids: Tensor,
        attention: bool = False,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate_single(
                input_ids=input_ids,
                max_new_length=input_ids.shape[1] + 1,
                temperature=1.0,
                top_k=0,
                top_p=1,
                attention=attention,
                num_to_masks=num_to_masks,
            )

        next_token_probs = out[-1].detach().cpu().numpy()

        return next_token_probs

    def clean(self) -> None:
        pass


def get_model_interface(
    model_arch: MODEL_ARCH, model_size: str, device: Optional[torch.device] = None
) -> ModelInterface:
    match model_arch:
        case MODEL_ARCH.MINIMAL_MAMBA2_new:
            return Mamba2Interface(model_size, device)
        case MODEL_ARCH.MAMBA1:
            return Mamba1Interface(model_size, device)
        case _:
            assert_never(model_arch)
