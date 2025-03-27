import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional, TypedDict

import numpy as np
import torch
from cachetools import TTLCache, cached
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
    BaseVariantParams,
)
from src.experiments.infrastructure.model_interface import ModelInterface
from src.experiments.runners.evaluate_model import EvaluateModelConfig, EvaluateModelParams
from src.utils.infra.output_path import OutputKey

# Time in seconds between intermediate saves
SAVE_INTERVAL = 600  # 10 minutes
PRINT_INTERVAL = 100

TWindowLayerStartIndex = tuple[TLayerIndex, ...]


class InfoFlowJSONFileCols:
    data: Literal["data"] = "data"
    metadata: Literal["metadata"] = "metadata"


class InfoFlowJSONMetadataCols:
    layers_amount: Literal["layers_amount"] = "layers_amount"
    banned_prompts: Literal["banned_prompts"] = "banned_prompts"


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
class InfoFlowFileStatistics:
    complete_prompt_ids: set[TPromptOriginalIndex]
    partial_prompt_ids: dict[TPromptOriginalIndex, set[TLayerIndex]]
    banned_prompt_ids: set[TPromptOriginalIndex]
    layers_amount: TLayerIndex

    @classmethod
    def from_json(cls, json_str: str) -> "InfoFlowFileStatistics":
        data = json.loads(json_str)
        return cls(
            complete_prompt_ids=set(data.pop("complete_prompt_ids")),
            partial_prompt_ids={
                TPromptOriginalIndex(int(prompt_id)): set(layer_ids)
                for prompt_id, layer_ids in data.pop("partial_prompt_ids").items()
            },
            banned_prompt_ids=set(data.pop("banned_prompt_ids")),
            layers_amount=data.pop("layers_amount"),
        )

    def to_json(self) -> str:
        data = asdict(self)
        data["partial_prompt_ids"] = {
            prompt_id: list(layer_ids) for prompt_id, layer_ids in data["partial_prompt_ids"].items()
        }
        data["banned_prompt_ids"] = list(data["banned_prompt_ids"])
        data["complete_prompt_ids"] = list(data["complete_prompt_ids"])
        return json.dumps(data, indent=4)


@dataclass(frozen=True)
class JSONInfoFlowFile:
    path: Path

    @property
    def statistics_path(self) -> Path:
        return self.path.with_suffix(".stats.json")

    def create_new(self, layers_amount: int) -> None:
        self.save(
            InfoFlowFileContent(
                metadata=InfoFlowMetadata(layers_amount=layers_amount, banned_prompts={}),
                data={},
            ),
        )

    def save(self, data: InfoFlowFileContent) -> None:
        self.statistics_path.unlink(missing_ok=True)
        self.path.write_text(json.dumps(data, indent=4))

    @cached(TTLCache(maxsize=1, ttl=60))
    def load(self) -> InfoFlowFileContent:
        if not self.path.exists():
            return InfoFlowFileContent(
                metadata=InfoFlowMetadata(layers_amount=0, banned_prompts={}),
                data={},
            )
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
        layer_idx_subset: Optional[TWindowLayerStartIndex] = None,
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
            else list(layer_idx_subset)
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

    @cached(TTLCache(maxsize=1, ttl=60))
    def get_statistics(self) -> InfoFlowFileStatistics:
        if self.statistics_path.exists():
            try:
                return InfoFlowFileStatistics.from_json(self.statistics_path.read_text())
            except Exception as e:
                print(f"Error reading statistics file at {self.statistics_path}: {e}")

        content = self.load()

        layers_amount = content[InfoFlowJSONFileCols.metadata][InfoFlowJSONMetadataCols.layers_amount]
        complete_prompt_ids: set[TPromptOriginalIndex] = set()
        partial_prompt_ids: dict[TPromptOriginalIndex, set[TLayerIndex]] = {}

        for prompt_id in content[InfoFlowJSONFileCols.data]:
            completed_layer_ids = set(content[InfoFlowJSONFileCols.data][prompt_id].keys())
            if len(completed_layer_ids) == layers_amount:
                complete_prompt_ids.add(prompt_id)
            else:
                partial_prompt_ids[prompt_id] = completed_layer_ids

        res = InfoFlowFileStatistics(
            complete_prompt_ids=complete_prompt_ids,
            partial_prompt_ids=partial_prompt_ids,
            banned_prompt_ids=set(
                content[InfoFlowJSONFileCols.metadata][InfoFlowJSONMetadataCols.banned_prompts].keys()
            ),
            layers_amount=layers_amount,
        )

        self.statistics_path.parent.mkdir(parents=True, exist_ok=True)
        self.statistics_path.write_text(res.to_json())
        return self.get_statistics()

    def get_banned_prompt_indices(self) -> set[TPromptOriginalIndex]:
        return set(self.get_statistics().banned_prompt_ids)

    def get_computed_prompt_idx(
        self, layer_idx_subset: Optional[TWindowLayerStartIndex] = None, include_banned: bool = False
    ) -> set[TPromptOriginalIndex]:
        statistics = self.get_statistics()
        ids = statistics.complete_prompt_ids
        if include_banned:
            ids.update(statistics.banned_prompt_ids)

        if layer_idx_subset is not None:
            layer_idx_set = set(layer_idx_subset)
            ids.update(
                prompt_id
                for prompt_id, layer_ids in statistics.partial_prompt_ids.items()
                if layer_idx_set.issubset(layer_ids)
            )

        return ids

    def get_missing_prompt_layer_values(
        self,
        prompt_idx_subset: list[TPromptOriginalIndex],
        layer_idx_subset: Optional[TWindowLayerStartIndex],
    ) -> dict[TPromptOriginalIndex, list[TLayerIndex]]:
        statistics = self.get_statistics()
        complete_prompt_ids = self.get_computed_prompt_idx(layer_idx_subset=layer_idx_subset, include_banned=True)
        missing_prompt_ids = {}
        full_layer_idx_subset = set(range(statistics.layers_amount))
        for prompt_id in prompt_idx_subset:
            if prompt_id not in complete_prompt_ids:
                if prompt_id in statistics.partial_prompt_ids:
                    missing_prompt_ids[prompt_id] = list(
                        full_layer_idx_subset - statistics.partial_prompt_ids[prompt_id]
                    )
                else:
                    missing_prompt_ids[prompt_id] = list(full_layer_idx_subset)
        return missing_prompt_ids


@dataclass(frozen=True)
class InfoFlowParams(BaseVariantParams):
    experiment_name: EXPERIMENT_NAMES = field(init=False, default=EXPERIMENT_NAMES.INFO_FLOW)
    window_size: TWindowSize
    source: TokenType
    feature_category: FeatureCategory
    target: TokenType
    subset_layers: Optional[TWindowLayerStartIndex] = None


class InfoFlowDependencies(TypedDict):
    evaluate_model: EvaluateModelConfig


@dataclass(frozen=True)
class InfoFlowRunner(BaseRunner[InfoFlowParams]):
    """Configuration for information flow analysis."""

    variant_params: InfoFlowParams

    @staticmethod
    def _get_variant_params():
        return InfoFlowParams

    @staticmethod
    def skip_task(model_arch: MODEL_ARCH, feature_category: FeatureCategory) -> bool:
        return not (is_mamba_arch(model_arch) or feature_category == FeatureCategory.ALL)

    def should_skip_task(self) -> bool:
        return self.skip_task(self.variant_params.model_arch, self.variant_params.feature_category)

    @classmethod
    def get_variant_output_keys(cls):
        return super().get_variant_output_keys() + [
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
        if isinstance(self.input_params.filteration, AnyExistingCompletePromptFilteration):
            prompt_ids = list(self.output_file.get_computed_prompt_idx())
        elif isinstance(self.input_params.filteration, AnyExistingPromptFilteration):
            prompt_ids = None
        else:
            prompt_ids = self.input_params.filteration.get_prompt_ids()
        return self.output_file.load_to_info_flow_output(
            prompt_idx_subset=prompt_ids,
        )

    def _compute_impl(self) -> None:
        run(self)

    def is_computed(self) -> bool:
        if not self.output_file.path.exists():
            return False
        return (
            len(
                self.output_file.get_missing_prompt_layer_values(
                    prompt_idx_subset=self.input_params.filteration.get_prompt_ids(),
                    layer_idx_subset=self.variant_params.subset_layers,
                )
            )
            == 0
        )

    def get_runner_dependencies(self) -> InfoFlowDependencies:  # type: ignore
        return InfoFlowDependencies(
            evaluate_model=EvaluateModelConfig.init_from_runner(
                self,
                variant_params=EvaluateModelParams(
                    model_arch=self.variant_params.model_arch,
                    model_size=self.variant_params.model_size,
                ),
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


def run(args: InfoFlowRunner):
    print(args)
    args.create_experiment_dir()

    model_interface = args.variant_params.get_model_interface()
    tokenizer = model_interface.tokenizer
    device = model_interface.device
    layers_amount = model_interface.n_layers() - args.variant_params.window_size + 1

    windows: dict[TLayerIndex, TWindow] = {
        layer_idx: TWindow(list(range(layer_idx, layer_idx + args.variant_params.window_size)))
        for layer_idx in range(0, layers_amount)
    }

    if not args.output_file.path.exists():
        args.output_file.create_new(layers_amount)

    missing_prompt_layer_values = args.output_file.get_missing_prompt_layer_values(
        prompt_idx_subset=args.input_params.filteration.get_prompt_ids(),
        layer_idx_subset=args.variant_params.subset_layers,
    )

    if not missing_prompt_layer_values:
        print("All outputs already exist")
        return

    content = args.output_file.load()

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
                    args.variant_params.source,
                    args.variant_params.feature_category,
                    args.variant_params.target,
                    model_interface,
                    tokenizer,
                    device,
                )
            except Exception as e:
                if "Test failure" in str(e):
                    # For test: Test failure is expected, so we raise the error
                    raise e
                print(f" Error evaluating {prompt_id = }: {e}")
                content[InfoFlowJSONFileCols.metadata][InfoFlowJSONMetadataCols.banned_prompts][prompt_id] = str(e)
                break

            current_time = time.time()
            if current_time - last_save_time >= SAVE_INTERVAL:
                args.output_file.save(content)
                last_save_time = current_time
                print(f"\nSaved intermediate results at prompt {prompt_id} and layer {layer_idx}")

    args.output_file.save(content)
