from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import MambaForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

import src.models.minimal_mamba2_new as minimal_mamba2_new
from src.consts import is_falcon
from src.knockout.attention_knockout.ssm_interfere import SSMInterfereHook
from src.knockout.knockout_mode import KnockoutMode
from src.types import MODEL_ARCH
from src.utils.setup_models import get_tokenizer_and_model


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
        self.tokenizer, self.model = get_tokenizer_and_model(model_arch, model_size, device)
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.device = self.model.device

    def setup(self, layers: Optional[Iterable[int]] = None):
        self.model.eval()

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


class Mamba1Interface(ModelInterface):
    model: MambaForCausalLM

    def __init__(
        self,
        model_size: str,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        is_falcon: bool = False,
    ):
        super().__init__(MODEL_ARCH.MAMBA1, model_size, device, tokenizer)

        self.hooks: list[SSMInterfereHook] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.is_falcon = is_falcon
        if is_falcon:
            print("using falcon")
        else:
            print("not using falcon")

        self.knockout_mode = KnockoutMode.ZERO_ATTENTION

    def setup(self, layers: Optional[Iterable[int]] = None):
        super().setup(layers)

        for handle in self.handles:
            handle.remove()

        # Assert that no hooks are left
        for m in self.model.modules():
            assert len(list(m._forward_hooks.items())) == 0

        self.handles = []
        self.hooks = []

        if layers is not None:
            # set up hooks
            for i in range(len(self.model.backbone.layers)):
                if i in layers:
                    # "mixer of interest" - moi
                    moi = self.model.backbone.layers[i].mixer

                    self.hooks.append(SSMInterfereHook(i, self.knockout_mode, is_falcon=self.is_falcon))

                    self.handles.append(moi.register_forward_hook(self.hooks[-1]))

    def generate_logits(
        self,
        input_ids: Tensor,
        attention: bool = False,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        if num_to_masks is not None:
            source_indices = []
            target_indices = []

            for layer in num_to_masks:
                source_indices = [num_to_masks[layer][i][1] for i in range(len(num_to_masks[layer]))]
                target_indices = [num_to_masks[layer][i][0] for i in range(len(num_to_masks[layer]))]
                break

            for hook in self.hooks:
                hook.knockout_indices = source_indices
                hook.affected_outputs = target_indices

        with torch.no_grad():
            out = self.model(input_ids)

        logits = out.logits
        probs = F.softmax(logits, dim=-1)

        return probs[:, -1, :].detach().cpu().numpy()  # type: ignore


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
        self.setup(num_to_masks)
        with torch.no_grad():
            out = self.model.generate_single(
                input_ids=input_ids,  # type: ignore
                max_new_length=input_ids.shape[1] + 1,
                temperature=1.0,
                top_k=0,
                top_p=1,
                attention=attention,
                num_to_masks=num_to_masks,
            )

        return out[-1].detach().cpu().numpy()  # type: ignore


def get_model_interface(
    model_arch: MODEL_ARCH, model_size: str, device: Optional[torch.device] = None
) -> ModelInterface:
    match model_arch:
        case MODEL_ARCH.MINIMAL_MAMBA2_new:
            return Mamba2Interface(model_size, device)
        case MODEL_ARCH.MAMBA1:
            return Mamba1Interface(model_size, device, is_falcon=is_falcon(model_size))
        case _:
            # assert_never(model_arch)
            raise ValueError(f"Unknown model architecture: {model_arch}")
