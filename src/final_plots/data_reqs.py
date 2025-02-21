import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union, cast

import pandas as pd

from src.consts import (
    COLUMNS,
    EXPERIMENT_NAMES,
    GRAPHS_ORDER,
    MODEL_ARCH,
    MODEL_SIZE_CAT,
    PATHS,
    TokenType,
    is_falcon,
    is_mamba_arch,
)
from src.experiments.evaluate_model import EvaluateModelConfig
from src.experiments.heatmap import HeatmapConfig
from src.experiments.info_flow import InfoFlowConfig
from src.final_plots.results_bank import HeatmapRecord, InfoFlowRecord, ResultRecord, get_experiment_results_bank
from src.types import MODEL_ARCH_AND_SIZE, FeatureCategory, TInfoFlowSource


class DataReq(NamedTuple):
    experiment_name: EXPERIMENT_NAMES
    model_arch: MODEL_ARCH
    model_size: str
    window_size: int
    is_all_correct: bool
    source: Optional[TokenType]
    feature_category: Optional[FeatureCategory]
    target: Optional[TokenType]
    prompt_idx: Optional[int]

    def get_config(self, variation: Optional[str] = None) -> Union[InfoFlowConfig, HeatmapConfig]:
        assert not self.is_all_correct

        if self.experiment_name == EXPERIMENT_NAMES.INFO_FLOW:
            assert self.source is not None
            assert self.target is not None
            token_source: TInfoFlowSource = (
                self.source if self.feature_category is None else (self.source, self.feature_category)
            )
            config = InfoFlowConfig(
                model_arch=self.model_arch,
                model_size=self.model_size,
                window_size=self.window_size,
                knockout_map={
                    self.target: [token_source],
                },
            )
        elif self.experiment_name == EXPERIMENT_NAMES.HEATMAP:
            assert self.prompt_idx is not None
            config = HeatmapConfig(
                model_arch=self.model_arch,
                model_size=self.model_size,
                window_size=self.window_size,
                prompt_indices_rows=[],
                prompt_original_indices=[self.prompt_idx],
            )
        else:
            raise ValueError(f"Unknown experiment name: {self.experiment_name}")

        if variation is not None:
            config.variation = variation
        return config


IDataFulfilled = dict[DataReq, Optional[Path]]
IDataFulfilledOptions = dict[DataReq, list[Path]]

DATA_FULFILLED_OVERIDES_PATH = Path(__file__).parent / "data_fulfilled_overides.csv"
DATA_FULFILLED_PATH = Path(__file__).parent / "data_fulfilled.csv"
PROMPT_SELECTION_PATH = Path(__file__).parent / "prompt_selections.json"


def result_record_to_data_req(result_record: ResultRecord) -> DataReq:
    if isinstance(result_record, InfoFlowRecord):
        target = result_record.target
        feature_category = result_record.feature_category
        source = result_record.source
        prompt_idx = None
    elif isinstance(result_record, HeatmapRecord):
        target = None
        feature_category = None
        source = None
        prompt_idx = result_record.prompt_idx
    else:
        raise ValueError(f"Unknown result record type: {type(result_record)}")
    return DataReq(
        experiment_name=result_record.experiment_name,
        model_arch=result_record.model_arch,
        model_size=result_record.model_size,
        window_size=result_record.window_size,
        is_all_correct=result_record.is_all_correct,
        source=source,
        feature_category=feature_category,
        target=target,
        prompt_idx=prompt_idx,
    )


def get_data_fullfment_options() -> IDataFulfilledOptions:
    data_reqs = get_data_reqs()
    data_reqs_options: IDataFulfilledOptions = {data_req: [] for data_req in data_reqs}
    results = get_experiment_results_bank()
    for result in results:
        data_req = result_record_to_data_req(result)
        if data_req in data_reqs_options:
            data_reqs_options[data_req].append(result.path)
    return data_reqs_options


def merge_data_reqs(first: IDataFulfilled, second: IDataFulfilled, keys_by_first: bool = True) -> IDataFulfilled:
    return {data_req: first.get(data_req) or second.get(data_req) for data_req in (first if keys_by_first else second)}


def get_latest_data_fulfilled() -> IDataFulfilled:
    data_reqs_options = get_data_fullfment_options()
    return {data_req: max(options) if options else None for data_req, options in data_reqs_options.items()}


def _save_data_fulfilled(path: Path, data_fulfilled: IDataFulfilled) -> None:
    if not data_fulfilled:
        if path.exists():
            path.unlink()
        return
    (
        pd.DataFrame.from_records(
            [
                {
                    **data_req._asdict(),
                    "path": None if path is None else path.relative_to(PATHS.PROJECT_DIR),
                }
                for data_req, path in data_fulfilled.items()
            ]
        ).to_csv(path, index=False)
    )


def save_data_fulfilled_overides(data_fulfilled: IDataFulfilled) -> None:
    _save_data_fulfilled(DATA_FULFILLED_OVERIDES_PATH, data_fulfilled)


def _load_data_fulfilled(path: Path) -> IDataFulfilled:
    if not path.exists():
        return {}
    return cast(
        IDataFulfilled,
        {DataReq(**row): row["path"] for row in pd.read_csv(path).to_records(index=False)},
    )


def load_data_fulfilled_overides() -> IDataFulfilled:
    return _load_data_fulfilled(DATA_FULFILLED_OVERIDES_PATH)


# region Add data reqs
STANDARD_WINDOW_SIZE_FOR_INFO_FLOW = 9
STANDARD_WINDOW_SIZE_FOR_HEATMAP = 5
ALL_WINDOW_SIZES = [1, 3, 5, 9, 12, 15]


def get_data_reqs() -> IDataFulfilled:
    data_reqs: IDataFulfilled = {}

    # region 1. Figure 1 Knockout information flow to the **last** token.
    """
    Figure 1: Knockout information flow to the **last** token.
    1. 4 subplots
    2. Columns - normalized change in [probability \\ accuracy]
    3. Rows        - Comparison between [Mamba1 2.8B \\ Mamba2 2.8B] and GPT2 1.5B
    4. Different colours indicate different source for knockout
    5. Trend shape indicate the model (solid for Mamba, dots for GPT)
        
    model sizes = ALL
    model archs = ALL
    window sizes = [STANDARD_WINDOW_SIZE_FOR_INFO_FLOW]
    is_all_correct = [True, False]
    feature_category = [None]
    target = [last]
    source = [last, first, subject, relation]
    """

    for model_arch_and_size in GRAPHS_ORDER:
        for is_all_correct in [True, False]:
            for source in [TokenType.last, TokenType.first, TokenType.subject, TokenType.relation]:
                data_reqs[
                    DataReq(
                        experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                        model_arch=model_arch_and_size.arch,
                        model_size=model_arch_and_size.size,
                        window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                        is_all_correct=is_all_correct,
                        source=source,
                        feature_category=None,
                        target=TokenType.last,
                        prompt_idx=None,
                    )
                ] = None

    # endregion

    # region 2. Figure 2 Knockout information flow to the last token - comparing model sizes.
    """
    Figure 2: Knockout information flow to the last token - comparing model sizes:
        1. 6 subplots
        2. Only normalized change in probability
        3. Columns - mamba [1 \\ 2]
        4. Rows        - model sizes
    model sizes = ALL
    model archs = [Mamba1, Mamba2,]
    window sizes = [STANDARD_WINDOW_SIZE_FOR_INFO_FLOW]
    is_all_correct = [True, False]
    target = [last]
    source = [last, first, subject, relation, subject-SLOW_DECAY, subject-FAST_DECAY]
    """

    for model_arch_and_size in GRAPHS_ORDER:
        if is_mamba_arch(model_arch_and_size.arch):
            for is_all_correct in [True, False]:
                for source, feature_category in [
                    (TokenType.last, None),
                    (TokenType.first, None),
                    (TokenType.subject, None),
                    (TokenType.relation, None),
                    (TokenType.subject, FeatureCategory.SLOW_DECAY),
                    (TokenType.subject, FeatureCategory.FAST_DECAY),
                ]:
                    data_reqs[
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                            model_arch=model_arch_and_size.arch,
                            model_size=model_arch_and_size.size,
                            window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                            is_all_correct=is_all_correct,
                            source=source,
                            feature_category=feature_category,
                            target=TokenType.last,
                            prompt_idx=None,
                        )
                    ] = None

    # endregion

    # region 3. Figure 3 Knockout information flow to the last token - Falcon Mamba 7B.
    """
    Figure 3: Knockout information flow to the last token - Falcon Mamba 7B:
    model sizes = [HUGE]
    model archs = [Falcon Mamba]
    window sizes = [STANDARD_WINDOW_SIZE_FOR_INFO_FLOW]
    is_all_correct = [True]
    target = [last]
    source = [last, first, subject, relation, subject-SLOW_DECAY, subject-FAST_DECAY]
    """

    for model_arch_and_size in GRAPHS_ORDER:
        if is_falcon(model_arch_and_size.size):
            for source, feature_category in [
                (TokenType.last, None),
                (TokenType.first, None),
                (TokenType.subject, None),
                (TokenType.relation, None),
                (TokenType.subject, FeatureCategory.SLOW_DECAY),
                (TokenType.subject, FeatureCategory.FAST_DECAY),
            ]:
                data_reqs[
                    DataReq(
                        experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                        model_arch=model_arch_and_size.arch,
                        model_size=model_arch_and_size.size,
                        window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                        is_all_correct=True,
                        source=source,
                        feature_category=feature_category,
                        target=TokenType.last,
                        prompt_idx=None,
                    )
                ] = None

    # endregion

    # region 4. Figure 4 Knockout information flow to the subject tokens.
    """
    Figure 4: Knockout information flow to the subject tokens:
        1. 4 subplots
        2. Columns - normalized change in [probability \\ accuracy]
        3. Rows        - Comparison between [Mamba1 2.8B \\ Mamba2 2.8B] and GPT2 1.5B
        4. Different colours indicate different source for knockout
        5. Trend shape indicate the model (solid for Mamba, dots for GPT)
    model archs = [Mamba1, Mamba2, GPT2]
    model sizes = [SMALL, MEDIUM, LARGE, HUGE]
    window sizes = [STANDARD_WINDOW_SIZE_FOR_INFO_FLOW]
    is_all_correct = [True, False]
    target = [subject]
    source = [context, subject]
    """

    for model_arch_and_size, model_size_cat in GRAPHS_ORDER.items():
        if model_size_cat != MODEL_SIZE_CAT.LARGE:
            continue
        for is_all_correct in [True, False]:
            for source in [TokenType.context, TokenType.subject]:
                data_reqs[
                    DataReq(
                        experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                        model_arch=model_arch_and_size.arch,
                        model_size=model_arch_and_size.size,
                        window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                        is_all_correct=is_all_correct,
                        source=source,
                        feature_category=None,
                        target=TokenType.subject,
                        prompt_idx=None,
                    )
                ] = None

    # endregion

    # region 5. Figure 5 Feature knockout. 4 subplots:
    """
    Figure 5: Feature knockout. 4 subplots:
        1. Columns - normalized change in [probability \\ accuracy]
        2. Rows        - knock out features with [high \\ low] norm
        3. Knocking out features with low norm
        4. In each subplots we have the results for Mamba1 2.8B, Mamba2 2.8B, Falcon Mamba 7B
        5. Trend color indicate the model in question.
        6. Trend shape indicates if this is when knocking out all features,
            or only subset of features.model archs = [Mamba1, Mamba2]
    model sizes = ALL
    window sizes = [STANDARD_WINDOW_SIZE_FOR_INFO_FLOW]
    is_all_correct = [True, False]
    target = [last]
    source = [context, subject, relation, subject-SLOW_DECAY, subject-FAST_DECAY]
    """

    for model_arch_and_size in GRAPHS_ORDER:
        if is_mamba_arch(model_arch_and_size.arch):
            for is_all_correct in [True, False]:
                for source, feature_category in [
                    (TokenType.last, None),
                    (TokenType.subject, FeatureCategory.SLOW_DECAY),
                    (TokenType.subject, FeatureCategory.FAST_DECAY),
                    (TokenType.first, None),
                    (TokenType.subject, None),
                    (TokenType.relation, None),
                ]:
                    data_reqs[
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                            model_arch=model_arch_and_size.arch,
                            model_size=model_arch_and_size.size,
                            window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                            is_all_correct=is_all_correct,
                            source=source,
                            feature_category=feature_category,
                            target=TokenType.last,
                            prompt_idx=None,
                        )
                    ] = None

    # endregion

    # also for 'last' target with all vars of decay only for mamba1 & 2 of the large sizes

    for source in [TokenType.relation, TokenType.first, TokenType.last, TokenType.all]:
        for model_arch_and_size, model_size_cat in GRAPHS_ORDER.items():
            if is_mamba_arch(model_arch_and_size.arch) and model_size_cat == MODEL_SIZE_CAT.LARGE:
                for feature_category in [FeatureCategory.SLOW_DECAY, FeatureCategory.FAST_DECAY]:
                    data_reqs[
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                            model_arch=model_arch_and_size.arch,
                            model_size=model_arch_and_size.size,
                            window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                            is_all_correct=False,
                            source=source,
                            feature_category=feature_category,
                            target=TokenType.last,
                            prompt_idx=None,
                        )
                    ] = None

    # region 6. Figure 6 Heatmaps.
    """
    Figure 6: Heatmaps. 3 subplots:
        1. Different columns are different models (Mamba 1 \\ Mamba 2 \\ GPT2.
    model archs = [Mamba1, Mamba2, GPT2]
    model sizes = ALL
    window sizes = [STANDARD_WINDOW_SIZE_FOR_HEATMAP]
    """
    # TODO: Add data reqs
    # endregion

    # region 7. Appendix: Window size.
    """
    App Figure 1: Window size:
        1. A table of subplots:
            1. Each row indicates the window size
            2. Each column indicate the architecture (Mamba 1 \\ 2)
        2. A different table per model size (130M \\ 1.4B \\ 2.8B)
    model archs = [Mamba1, Mamba2]
    model sizes = ALL
    window sizes = [ALL]
    is_all_correct = [True]
    target = [last]
    source = [last, first, subject, relation]

    """
    for model_arch_and_size in GRAPHS_ORDER:
        if is_mamba_arch(model_arch_and_size.arch):
            for window_size in ALL_WINDOW_SIZES:
                for source in [TokenType.last, TokenType.first, TokenType.subject, TokenType.relation]:
                    data_reqs[
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                            model_arch=model_arch_and_size.arch,
                            model_size=model_arch_and_size.size,
                            window_size=window_size,
                            is_all_correct=True,
                            source=source,
                            feature_category=None,
                            target=TokenType.last,
                            prompt_idx=None,
                        )
                    ] = None

    # endregion

    # region 8. Appendix: Knockout information flow to the last token - comparing model sizes.
    """
    App Figure 2: Knockout information flow to the last token - comparing model sizes:
        1. 6 subplots
        2. Only normalized change in accuracy
        3. Columns - mamba 1 \\ 2
        4. Rows        - model sizes
    model archs = [Mamba1, Mamba2]
    model sizes = [SMALL, MEDIUM, LARGE, HUGE]
    window sizes = [STANDARD_WINDOW_SIZE_FOR_INFO_FLOW]
    is_all_correct = [True, False]
    target = [last]
    source = [last, first, subject, relation]
    """

    for model_arch_and_size in GRAPHS_ORDER:
        if is_mamba_arch(model_arch_and_size.arch):
            for is_all_correct in [True, False]:
                for source in [TokenType.last, TokenType.first, TokenType.subject, TokenType.relation]:
                    data_reqs[
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                            model_arch=model_arch_and_size.arch,
                            model_size=model_arch_and_size.size,
                            window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                            is_all_correct=is_all_correct,
                            source=source,
                            feature_category=None,
                            target=TokenType.last,
                            prompt_idx=None,
                        )
                    ] = None

    # endregion

    # region 9. Appendix: Heatmaps.
    """
    App Figure 3: Heatmaps
        1. A different plot per (model arch, model size, window size)
        2. Each subplot is per example
    model archs = [Mamba1, Mamba2, GPT2]
    model sizes = [SMALL, MEDIUM, LARGE, HUGE]
    window sizes = [STANDARD_WINDOW_SIZE_FOR_HEATMAP]
    """
    # TODO: Add data reqs
    # endregion

    return data_reqs


def get_current_data_reqs() -> IDataFulfilled:
    data_reqs_options = get_latest_data_fulfilled()
    loaded_overides: dict[DataReq, Path | None] = load_data_fulfilled_overides()
    return merge_data_reqs(loaded_overides, data_reqs_options, keys_by_first=False)


def update_data_reqs_with_latest_results() -> None:
    _save_data_fulfilled(DATA_FULFILLED_PATH, get_current_data_reqs())


def save_prompt_selections(prompt_selections: list[tuple[set[MODEL_ARCH_AND_SIZE], int]]) -> None:
    pd.DataFrame(prompt_selections).to_csv(PROMPT_SELECTION_PATH, index=False)


def load_prompt_selections() -> list[tuple[set[MODEL_ARCH_AND_SIZE], int]]:
    return pd.read_csv(PROMPT_SELECTION_PATH).to_dict(orient="records")  # type: ignore


def get_model_evaluations(
    variation: str, model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
) -> dict[MODEL_ARCH_AND_SIZE, pd.DataFrame]:
    return {
        model_arch_and_size: EvaluateModelConfig(
            model_arch=model_arch_and_size[0],
            model_size=model_arch_and_size[1],
            variation=variation,
        )
        .get_outputs()
        .set_index(COLUMNS.ORIGINAL_IDX)
        for model_arch_and_size in model_arch_and_sizes
    }


@dataclass
class ModelCombination:
    correct_models: set[MODEL_ARCH_AND_SIZE]
    incorrect_models: set[MODEL_ARCH_AND_SIZE]
    prompts: list[int]
    chosen_prompt: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct_models": list(self.correct_models),
            "incorrect_models": list(self.incorrect_models),
            "prompts": self.prompts,
            "chosen_prompt": self.chosen_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCombination":
        return cls(
            correct_models=set(data["correct_models"]),
            incorrect_models=set(data["incorrect_models"]),
            prompts=data["prompts"],
            chosen_prompt=data["chosen_prompt"],
        )


def save_model_combinations_prompts(model_combinations: list[ModelCombination]) -> None:
    json.dump(
        [model_combination.to_dict() for model_combination in model_combinations],
        PROMPT_SELECTION_PATH.open("w"),
    )


def get_model_combinations_prompts(
    variation: Optional[str], model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE], seed: Optional[int] = 42
) -> list[ModelCombination]:
    """Get all possible model combinations and their corresponding prompts.
    Each combination specifies which models should be correct and which should be incorrect.

    Returns:
        List of ModelCombination objects describing each valid combination pattern.
    """
    if PROMPT_SELECTION_PATH.exists():
        saved_selections = json.load(PROMPT_SELECTION_PATH.open("r"))
        saved_combinations = [ModelCombination.from_dict(selection) for selection in saved_selections]

        # Get all unique models from saved combinations
        saved_models = set()
        for combo in saved_combinations:
            saved_models.update(combo.correct_models)
            saved_models.update(combo.incorrect_models)

        # If requested models are a subset of saved models, derive from existing
        if saved_models - set(model_arch_and_sizes):
            return derive_subset_model_combinations(saved_combinations, model_arch_and_sizes)
        else:
            return saved_combinations

    # Otherwise generate new combinations
    assert seed is not None
    assert variation is not None
    model_evaluations = get_model_evaluations(variation, model_arch_and_sizes)
    # Get all prompts
    all_prompts = set(model_evaluations[model_arch_and_sizes[0]].index)

    # Create a DataFrame with correctness for each model
    correctness_df = pd.DataFrame(index=sorted(all_prompts))
    for model_arch_and_size in model_arch_and_sizes:
        model_df = model_evaluations[model_arch_and_size]
        correctness_df[model_arch_and_size] = [
            model_df.at[idx, COLUMNS.MODEL_CORRECT] if idx in model_df.index else False for idx in correctness_df.index
        ]

    # Generate all possible combinations
    combinations: list[ModelCombination] = []

    # Convert to numpy for faster operations
    correctness_matrix = correctness_df.values

    # For each possible combination of models being correct/incorrect
    for i in range(2 ** len(model_arch_and_sizes)):
        # Convert number to binary to get combination of correct models
        binary = format(i, f"0{len(model_arch_and_sizes)}b")

        # Get models that should be correct and incorrect
        correct_models = [model_arch_and_sizes[j] for j, bit in enumerate(binary) if bit == "1"]
        incorrect_models = [model_arch_and_sizes[j] for j, bit in enumerate(binary) if bit == "0"]

        # Find prompts that are correct for all correct_models AND incorrect for all incorrect_models
        correct_mask = correctness_matrix[:, [j for j, bit in enumerate(binary) if bit == "1"]].all(axis=1)
        incorrect_mask = ~correctness_matrix[:, [j for j, bit in enumerate(binary) if bit == "0"]].any(axis=1)

        # Combined mask for prompts meeting both conditions
        mask = correct_mask & incorrect_mask
        matching_prompts = correctness_df.index[mask].tolist()

        random.seed(seed)
        if matching_prompts:
            chosen_prompt = random.choice(matching_prompts)
        else:
            chosen_prompt = None

        combinations.append(
            ModelCombination(
                correct_models=set(correct_models),
                incorrect_models=set(incorrect_models),
                prompts=matching_prompts,
                chosen_prompt=chosen_prompt,
            )
        )

    save_model_combinations_prompts(combinations)
    return get_model_combinations_prompts(variation, model_arch_and_sizes, None)


def derive_subset_model_combinations(
    saved_combinations: list[ModelCombination], requested_models: list[MODEL_ARCH_AND_SIZE]
) -> list[ModelCombination]:
    """Derive model combinations for a subset using saved combinations with O(|C|) complexity."""
    requested_set = set(requested_models)
    pattern_map: dict[tuple[frozenset, frozenset], tuple[list[int], list[int]]] = {}

    # First pass: Group by projected patterns and collect prompts
    for combo in saved_combinations:
        # Project to subset - O(|S|) per combination
        proj_correct = frozenset(m for m in combo.correct_models if m in requested_set)
        proj_incorrect = frozenset(m for m in combo.incorrect_models if m in requested_set)

        # Get existing or create new entry
        key = (proj_correct, proj_incorrect)
        prompts, chosen_prompts = pattern_map.get(key, ([], []))

        # Extend with this combination's prompts
        prompts.extend(combo.prompts)
        if combo.chosen_prompt is not None:
            chosen_prompts.append(combo.chosen_prompt)

        pattern_map[key] = (prompts, chosen_prompts)

    # Second pass: Create new combinations
    new_combinations = []
    for (correct, incorrect), (prompts, chosen_prompts) in pattern_map.items():
        # Remove duplicate prompts while preserving order
        seen = set()
        unique_prompts = [p for p in prompts if not (p in seen or seen.add(p))]

        # Preserve chosen prompt order from original combinations
        chosen_candidates = [p for p in chosen_prompts if p in unique_prompts]
        chosen_prompt = chosen_candidates[0] if chosen_candidates else None

        new_combinations.append(
            ModelCombination(
                correct_models=set(correct),
                incorrect_models=set(incorrect),
                prompts=unique_prompts,
                chosen_prompt=chosen_prompt or (unique_prompts[0] if unique_prompts else None),
            )
        )

    return sorted(new_combinations, key=lambda x: (-len(x.prompts), x.chosen_prompt))
