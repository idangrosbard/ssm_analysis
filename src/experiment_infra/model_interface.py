from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, Union, assert_never

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import MambaForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

import src.models.minimal_mamba2 as minimal_mamba2
from src.consts import is_falcon
from src.knockout.attention_knockout import gpt2_knockout_utils
from src.knockout.attention_knockout.ssm_interfere import SSMInterfereHook
from src.types import MODEL_ARCH, FeatureCategory, KnockoutMode
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
        feature_category: FeatureCategory = FeatureCategory.ALL,
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
    def n_layers(self) -> int:
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

    def _get_feature_mask(self, layer: torch.nn.Module, feature_category: FeatureCategory) -> Tensor:
        assert isinstance(layer.A_log, torch.Tensor)

        if feature_category == FeatureCategory.ALL:
            return torch.zeros(layer.A_log.shape[0])

        if feature_category == FeatureCategory.NONE:
            return torch.ones(layer.A_log.shape[0])

        decay_matrices = torch.exp(-torch.exp(layer.A_log))
        n_ssms = decay_matrices.shape[0]

        # get the norms
        norms = torch.norm(decay_matrices, p=1, dim=1)

        sorted_indices = torch.argsort(norms, descending=(feature_category == FeatureCategory.SLOW_DECAY))
        mask = torch.zeros_like(norms, dtype=torch.bool)
        mask[sorted_indices[: n_ssms // 3]] = True
        return mask

    def generate_logits(
        self,
        input_ids: Tensor,
        attention: bool = False,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        feature_category: FeatureCategory = FeatureCategory.ALL,
    ) -> torch.Tensor:
        if num_to_masks is not None:
            source_indices = []
            target_indices = []

            for layer, hook in zip(num_to_masks, self.hooks):
                source_indices = [num_to_masks[layer][i][1] for i in range(len(num_to_masks[layer]))]
                target_indices = [num_to_masks[layer][i][0] for i in range(len(num_to_masks[layer]))]

                hook.knockout_indices = source_indices
                hook.affected_outputs = target_indices
                hook.feature_mask = self._get_feature_mask(self.model.backbone.layers[layer].mixer, feature_category)

        with torch.no_grad():
            out = self.model(input_ids)

        logits = out.logits
        probs = F.softmax(logits, dim=-1)

        return probs[:, -1, :].detach().cpu().numpy()  # type: ignore

    def n_layers(self) -> int:
        return len(self.model.backbone.layers)


class Mamba2Interface(ModelInterface):
    model: minimal_mamba2.Mamba2LMHeadModel

    def __init__(
        self,
        model_size: str,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        feature_category: Optional[FeatureCategory] = None,
    ):
        super().__init__(MODEL_ARCH.MAMBA2, model_size, device, tokenizer)

    def _get_feature_mask(self, layer: torch.nn.Module, feature_category: FeatureCategory) -> Tensor:
        assert isinstance(layer.A_log, torch.Tensor)

        if feature_category == FeatureCategory.ALL:
            return torch.zeros(layer.A_log.shape[0])

        if feature_category == FeatureCategory.NONE:
            return torch.ones(layer.A_log.shape[0])

        decay_matrices = torch.exp(-torch.exp(layer.A_log)).unsqueeze(-1)
        n_ssms = decay_matrices.shape[0]

        # get the norms
        norms = torch.norm(decay_matrices, p=1, dim=1)

        sorted_indices = torch.argsort(norms, descending=(feature_category == FeatureCategory.SLOW_DECAY))
        mask = torch.zeros_like(norms, dtype=torch.bool)
        mask[sorted_indices[: n_ssms // 3]] = True
        return mask

    def generate_logits(
        self,
        input_ids: Tensor,
        attention: bool = False,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        feature_category: FeatureCategory = FeatureCategory.ALL,
    ) -> torch.Tensor:
        self.setup(num_to_masks)

        feature_masks = {}
        if num_to_masks is not None:
            for layer in num_to_masks:
                feature_masks[layer] = self._get_feature_mask(
                    self.model.backbone.layers[layer].mixer, feature_category
                ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate_single(
                input_ids=input_ids,  # type: ignore
                max_new_length=input_ids.shape[1] + 1,
                temperature=1.0,
                top_k=0,
                top_p=1,
                attention=attention,
                num_to_masks=num_to_masks,
                feature_mask=feature_masks,
            )

        return out[-1].detach().cpu().numpy()  # type: ignore

    def n_layers(self) -> int:
        return len(self.model.backbone.layers)


class GPT2Interface(ModelInterface):
    model: MambaForCausalLM

    def __init__(
        self,
        model_size: str,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    ):
        super().__init__(MODEL_ARCH.GPT2, model_size, device, tokenizer)

        self.knockout_mode = KnockoutMode.ZERO_ATTENTION

    def _trace_with_attn_block(
        self,
        model,
        inp,
        from_to_index_per_layer,  # A list of (source index, target index) to block
    ):
        with torch.no_grad():
            # set hooks
            block_attn_hooks = gpt2_knockout_utils.set_block_attn_hooks(model, from_to_index_per_layer)

            # get prediction
            outputs_exp = model(**inp)

            # remove hooks
            gpt2_knockout_utils.remove_wrapper(model, block_attn_hooks)

        probs = torch.softmax(outputs_exp.logits[:, -1, :], dim=-1)

        return probs

    def generate_logits(
        self,
        input_ids: Tensor,
        attention: bool = False,
        num_to_masks: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        feature_category: FeatureCategory = FeatureCategory.ALL,
    ) -> torch.Tensor:
        assert input_ids.shape[0] == 1
        num_to_masks = num_to_masks or {}
        max_len = input_ids.shape[1]
        attention_mask = [[1] * max_len]
        inp = dict(
            input_ids=input_ids.to(self.model.device),
            attention_mask=torch.tensor(attention_mask).to(self.model.device),
        )

        probs = self._trace_with_attn_block(self.model, inp, num_to_masks)

        return probs.detach().cpu().numpy()  # type: ignore

    def n_layers(self) -> int:
        return len(self.model.transformer.h)


MODEL_INTERFACES_CACHE: dict[tuple[MODEL_ARCH, str], ModelInterface] = {}


def get_model_interface(
    model_arch: MODEL_ARCH, model_size: str, device: Optional[torch.device] = None
) -> ModelInterface:
    key = (model_arch, model_size)
    if key in MODEL_INTERFACES_CACHE:
        return MODEL_INTERFACES_CACHE[key]

    model_interface: Optional[ModelInterface] = None
    match model_arch:
        case MODEL_ARCH.MAMBA2:
            model_interface = Mamba2Interface(model_size, device)
        case MODEL_ARCH.MAMBA1:
            model_interface = Mamba1Interface(model_size, device, is_falcon=is_falcon(model_size))
        case MODEL_ARCH.GPT2:
            model_interface = GPT2Interface(model_size, device)
        case MODEL_ARCH.LLAMA2 | MODEL_ARCH.LLAMA3_2:
            raise NotImplementedError("LLama models are not supported yet")
        case _:
            assert_never(model_arch)

    MODEL_INTERFACES_CACHE[key] = model_interface
    return model_interface
