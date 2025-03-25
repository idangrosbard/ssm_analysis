import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import torch
from tqdm import tqdm

from src.analysis.prompt_filterations import (
    AllPromptFilteration,
    AnyExistingCompletePromptFilteration,
    AnyExistingPromptFilteration,
)
from src.core.consts import is_mamba_arch
from src.core.names import (
    COLS,
    DATASETS,
    EXPERIMENT_NAMES,
    INFO_FLOW_HP_COLS,
    InfoFlowCols,
    InfoFlowJSONFileCols,
    InfoFlowJSONMetadataCols,
)
from src.core.types import (
    MODEL_ARCH,
    FeatureCategory,
    TInfoFlowOutput,
    TInfoFlowWindowValue,
    TLayerIndex,
    TokenType,
    TPromptOriginalIndex,
    TTokenizer,
    TWindow,
    TWindowSize,
)
from src.data_ingestion.helpers.logits_utils import Prompt, get_num_to_masks, get_prompt_row_index
from src.experiments.infrastructure.base_config import (
    BASE_OUTPUT_KEYS,
    BaseRunner,
)
from src.experiments.infrastructure.model_interface import ModelInterface
from src.experiments.runners.evaluate_model import EvaluateModelConfig, EvaluateModelParams
from src.utils.infra.output_path import OutputKey

# Time in seconds between intermediate saves
SAVE_INTERVAL = 600  # 10 minutes
PRINT_INTERVAL = 100


def skip_task(model_arch: MODEL_ARCH, feature_category: FeatureCategory) -> bool:
    return not (is_mamba_arch(model_arch) or feature_category == FeatureCategory.ALL)


@dataclass
class InfoFlowParams:
    window_size: TWindowSize
    source: TokenType
    feature_category: FeatureCategory
    target: TokenType


class InfoFlowDependencies(TypedDict):
    evaluate_model: EvaluateModelConfig


class InfoFlowMetadata(TypedDict):
    layers_amount: TLayerIndex
    banned_prompts: dict[TPromptOriginalIndex, str]


class InfoFlowPromptLayerValue(TypedDict):
    hit: bool
    true_probs: float
    diffs: float


class InfoFlowFileContent(TypedDict):
    metadata: InfoFlowMetadata
    data: dict[TPromptOriginalIndex, dict[TLayerIndex, InfoFlowPromptLayerValue]]


@dataclass
class JSONInfoFlowFile:
    path: Path

    def create_new(self, layers_amount: int) -> None:
        self.save(
            InfoFlowFileContent(
                metadata=InfoFlowMetadata(layers_amount=layers_amount, banned_prompts={}),
                data={},
            ),
        )

    def save(self, data: InfoFlowFileContent) -> None:
        self.path.write_text(json.dumps(data, indent=4))

    def load(self) -> InfoFlowFileContent:
        raw_json = json.load(self.path.open("r"))
        return InfoFlowFileContent(
            metadata=InfoFlowMetadata(
                layers_amount=raw_json[InfoFlowJSONFileCols.metadata][InfoFlowJSONMetadataCols.layers_amount],
                banned_prompts={
                    TPromptOriginalIndex(int(prompt_id)): err_str
                    for prompt_id, err_str in raw_json[InfoFlowJSONFileCols.metadata][
                        InfoFlowJSONMetadataCols.banned_prompts
                    ].items()
                },
            ),
            data={
                TPromptOriginalIndex(int(prompt_id)): {
                    TLayerIndex(int(layer_id)): InfoFlowPromptLayerValue(
                        **raw_json[InfoFlowJSONFileCols.data][prompt_id][layer_id]
                    )
                    for layer_id in raw_json[InfoFlowJSONFileCols.data][prompt_id]
                }
                for prompt_id in raw_json[InfoFlowJSONFileCols.data]
            },
        )

    def load_to_info_flow_output(
        self,
        prompt_idx_subset: Optional[list[TPromptOriginalIndex]] = None,
        layer_idx_subset: Optional[list[TLayerIndex]] = None,
    ) -> TInfoFlowOutput:
        content = self.load()
        info_flow_data = content[InfoFlowJSONFileCols.data]

        prompt_idx: list[TPromptOriginalIndex] = (
            list(content[InfoFlowJSONFileCols.data].keys()) if prompt_idx_subset is None else prompt_idx_subset
        )

        # Preserve order for test output clarity
        # TODO: remove this after commiting tests results
        prompt_idx = [
            prompt_id
            for prompt_id in AllPromptFilteration(DATASETS.COUNTER_FACT).get_prompt_ids()
            if prompt_id in prompt_idx
        ]

        layer_idx: list[TLayerIndex] = (
            list(range(content[InfoFlowJSONFileCols.metadata][InfoFlowJSONMetadataCols.layers_amount]))
            if layer_idx_subset is None
            else layer_idx_subset
        )

        return {
            layer_id: TInfoFlowWindowValue(
                hit=[info_flow_data[prompt_idx][layer_id][InfoFlowCols.hit] for prompt_idx in prompt_idx],
                true_probs=[info_flow_data[prompt_idx][layer_id][InfoFlowCols.true_probs] for prompt_idx in prompt_idx],
                diffs=[info_flow_data[prompt_idx][layer_id][COLS.INFO_FLOW.DIFFS.value] for prompt_idx in prompt_idx],
                original_idx=prompt_idx,
            )
            for layer_id in layer_idx
        }

    def get_banned_prompt_indices(self) -> set[TPromptOriginalIndex]:
        return set(self.load()[InfoFlowJSONFileCols.metadata][InfoFlowJSONMetadataCols.banned_prompts])

    def get_existing_prompt_idx(
        self, layer_idx_subset: Optional[list[TLayerIndex]] = None
    ) -> list[TPromptOriginalIndex]:
        content: InfoFlowFileContent = self.load()
        info_flow_data = content["data"]

        layer_idx: list[TLayerIndex] = (
            list(range(content[InfoFlowJSONFileCols.metadata][InfoFlowJSONMetadataCols.layers_amount]))
            if layer_idx_subset is None
            else layer_idx_subset
        )

        return [
            prompt_id
            for prompt_id in info_flow_data.keys()
            if all(layer_id in info_flow_data[prompt_id] for layer_id in layer_idx)
        ]

    def get_missing_prompt_layer_values(
        self,
        prompt_idx_subset: Optional[list[TPromptOriginalIndex]] = None,
        layer_idx_subset: Optional[list[TLayerIndex]] = None,
    ) -> dict[TPromptOriginalIndex, list[TLayerIndex]]:
        content = self.load()
        info_flow_data = content["data"]

        prompt_idx: list[TPromptOriginalIndex] = (
            list(content[InfoFlowJSONFileCols.data].keys()) if prompt_idx_subset is None else prompt_idx_subset
        )

        layer_idx: list[TLayerIndex] = (
            list(range(content[InfoFlowJSONFileCols.metadata][InfoFlowJSONMetadataCols.layers_amount]))
            if layer_idx_subset is None
            else layer_idx_subset
        )

        res: dict[TPromptOriginalIndex, list[TLayerIndex]] = {}
        for prompt_id in prompt_idx:
            if prompt_id not in info_flow_data:
                res[prompt_id] = layer_idx
            elif all(layer_id not in info_flow_data[prompt_id] for layer_id in layer_idx):
                res[prompt_id] = layer_idx
        return res


@dataclass
class InfoFlowConfig(BaseRunner[InfoFlowParams, TInfoFlowOutput]):
    """Configuration for information flow analysis."""

    runner_params: InfoFlowParams

    @property
    def experiment_name(self):
        return EXPERIMENT_NAMES.INFO_FLOW

    @property
    def experiment_output_keys(self):
        return super().experiment_output_keys + [
            BASE_OUTPUT_KEYS.WINDOW_SIZE,
            OutputKey[TokenType](INFO_FLOW_HP_COLS.target),
            OutputKey[TokenType](INFO_FLOW_HP_COLS.source),
            OutputKey[FeatureCategory](INFO_FLOW_HP_COLS.feature_category),
        ]

    @property
    def output_file(self) -> JSONInfoFlowFile:
        return JSONInfoFlowFile(self.variation_paths.outputs_path / "info_flow.json")

    @staticmethod
    def load_output(path: Path) -> TInfoFlowOutput:
        return JSONInfoFlowFile(path).load_to_info_flow_output()

    def get_outputs(self) -> TInfoFlowOutput:
        if isinstance(self.prompt_filteration, AnyExistingCompletePromptFilteration):
            prompt_ids = self.output_file.get_existing_prompt_idx()
        elif isinstance(self.prompt_filteration, AnyExistingPromptFilteration):
            prompt_ids = None
        else:
            prompt_ids = self.prompt_ids
        return self.output_file.load_to_info_flow_output(
            prompt_idx_subset=prompt_ids,
        )

    def compute(self) -> None:
        run(self)

    def is_computed(self) -> bool:
        if not self.output_file.path.exists():
            return False
        return len(self.output_file.get_missing_prompt_layer_values(prompt_idx_subset=self.prompt_ids)) == 0

    def get_runner_dependencies(self) -> InfoFlowDependencies:  # type: ignore
        return InfoFlowDependencies(
            evaluate_model=EvaluateModelConfig.init_from_config(
                self,
                runner_params=EvaluateModelParams(),
            ),
        )


def forward_eval(
    prompt: Prompt,
    window: TWindow,
    knockout_source: TokenType,
    feature_category: FeatureCategory,
    knockout_target: TokenType,
    model_interface: ModelInterface,
    tokenizer: TTokenizer,
    device,
) -> InfoFlowPromptLayerValue:
    num_to_masks, first_token = get_num_to_masks(prompt, tokenizer, window, knockout_source, knockout_target, device)

    next_token_probs = model_interface.generate_logits(
        input_ids=prompt.input_ids(tokenizer, device),
        num_to_masks=num_to_masks,
        feature_category=feature_category,
    )

    max_prob = np.max(next_token_probs, axis=1)[0]
    true_id = prompt.true_id(tokenizer, "cpu")
    base_prob = prompt.base_prob
    true_prob = next_token_probs[0, true_id[:, 0]]
    torch.cuda.empty_cache()
    return {
        InfoFlowCols.hit: bool(true_prob == max_prob),
        InfoFlowCols.diffs: float(((true_prob - base_prob) / base_prob) * 100.0),
        # InfoFlowCols.first: first_token,
        # InfoFlowCols.diff_unnorm: true_prob - base_prob,
        InfoFlowCols.true_probs: float(true_prob),
    }


def run(args: InfoFlowConfig):
    print(args)
    args.create_experiment_run_path()

    model_interface = args.common_params.get_model_interface()
    tokenizer = model_interface.tokenizer
    device = model_interface.device
    layers_amount = model_interface.n_layers() - args.runner_params.window_size + 1

    windows: dict[TLayerIndex, TWindow] = {
        layer_idx: TWindow(list(range(layer_idx, layer_idx + args.runner_params.window_size)))
        for layer_idx in range(0, layers_amount)
    }

    if not args.output_file.path.exists():
        args.output_file.create_new(layers_amount)

    missing_prompt_layer_values = args.output_file.get_missing_prompt_layer_values(prompt_idx_subset=args.prompt_ids)
    content = args.output_file.load()

    if not missing_prompt_layer_values:
        print("All outputs already exist")
        return

    data = args.get_runner_dependencies()["evaluate_model"].get_prompt_data()

    last_save_time = time.time()
    missing_prompt_layer_values = sorted(missing_prompt_layer_values.items())
    for prompt_id, layer_idx in tqdm(
        missing_prompt_layer_values,
        desc="Missing prompts",
        total=len(missing_prompt_layer_values),
        mininterval=PRINT_INTERVAL,
    ):
        if prompt_id not in content[InfoFlowJSONFileCols.data]:
            content[InfoFlowJSONFileCols.data][prompt_id] = {}
        if prompt_id in content[InfoFlowJSONFileCols.metadata][InfoFlowJSONMetadataCols.banned_prompts]:
            continue
        for layer_idx in layer_idx:
            window = windows[layer_idx]
            model_interface.setup(layers=window)
            try:
                content[InfoFlowJSONFileCols.data][prompt_id][layer_idx] = forward_eval(
                    get_prompt_row_index(data, prompt_id),
                    window,
                    args.runner_params.source,
                    args.runner_params.feature_category,
                    args.runner_params.target,
                    model_interface,
                    tokenizer,
                    device,
                )
            except Exception as e:
                if "Test failure" in str(e):
                    # Test failure is expected, so we raise the error
                    raise e
                print(f" Error evaluating {prompt_id = }: {e}")
                content[InfoFlowJSONFileCols.metadata][InfoFlowJSONMetadataCols.banned_prompts][prompt_id] = str(e)
                continue

            current_time = time.time()
            if current_time - last_save_time >= SAVE_INTERVAL:
                args.output_file.save(content)
                last_save_time = current_time
                print(f"\nSaved intermediate results at prompt {prompt_id} and layer {layer_idx}")

    args.output_file.save(content)
