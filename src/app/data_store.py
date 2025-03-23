import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer
from streamlit import cache_resource

from src.analysis.experiment_results.data_requirements import (
    IDataFulfilled,
    ModelCombination,
    choose_latest_data_fulfilled,
    get_data_fullfment_options,
    get_data_reqs,
    get_model_combinations_prompts,
    get_model_evaluations,
)
from src.analysis.experiment_results.results_bank import (
    RESULTS_BASE_PATH,
    get_experiment_results_bank,
)
from src.app.app_consts import (
    GLOBAL_APP_CONSTS,
)
from src.core.names import DATASETS
from src.core.types import MODEL_ARCH_AND_SIZE, TPromptOriginalIndex, TVariationName, TWindowSize
from src.data_ingestion.data_defs import DataReqs, ResultBank, SummarizedDataFulfilledReqs
from src.experiments.infrastructure.base_config import CommonParams, SelectivePromptFilteration
from src.experiments.runners.heatmap import HeatmapConfig, HeatmapParams
from src.utils.streamlit.helpers.cache import CacheWithDependencies


@CacheWithDependencies()
def load_model_evaluations(variation: TVariationName, model_arch_and_size: MODEL_ARCH_AND_SIZE) -> pd.DataFrame:
    return get_model_evaluations(variation, [model_arch_and_size])[model_arch_and_size]


@CacheWithDependencies()
def load_model_evaluations_dict(variation: TVariationName) -> dict[MODEL_ARCH_AND_SIZE, pd.DataFrame]:
    """Load evaluation data for all models with caching"""
    return {
        model_arch_and_size: load_model_evaluations(variation, model_arch_and_size)
        for model_arch_and_size in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS
    }


@cache_resource
def merge_model_evaluations_streamlit_rendered(variation: TVariationName) -> StreamlitRenderer:
    """Load evaluation data for all models with caching"""
    return StreamlitRenderer(
        pd.concat(
            [
                df.assign(model_arch=key.arch, model_size=key.size)
                for key, df in load_model_evaluations_dict(variation).items()
            ]
        ),
        spec=f"model_evals_{variation}.csv",
        spec_io_mode="rw",
    )


@CacheWithDependencies()
def load_results_bank() -> ResultBank:
    return get_experiment_results_bank()


@CacheWithDependencies()
def load_test_results_bank() -> ResultBank:
    return get_experiment_results_bank(results_base_paths=(RESULTS_BASE_PATH.TEST,))


@CacheWithDependencies()
def load_data_reqs() -> DataReqs:
    return get_data_reqs()


@CacheWithDependencies()
def load_latest_fulfilled_reqs() -> IDataFulfilled:
    """Load the latest fulfilled requirements"""
    data_reqs_options = get_data_fullfment_options(load_data_reqs(), load_results_bank())
    return choose_latest_data_fulfilled(data_reqs_options)


# Data Requirements hooks
@CacheWithDependencies()
def load_fulfilled_reqs_df() -> SummarizedDataFulfilledReqs:
    """Load the data requirements options and overrides to dispaly the fulfilled requirements"""
    results_bank = load_results_bank()
    data_reqs = load_data_reqs()
    options = get_data_fullfment_options(data_reqs, results_bank)

    return SummarizedDataFulfilledReqs(options)


@CacheWithDependencies()
def get_merged_evaluations(prompt_idx: TPromptOriginalIndex, variation: TVariationName) -> pd.DataFrame:
    """Get merged evaluations for a specific prompt.

    Args:
        prompt_idx: The prompt index to get evaluations for

    Returns:
        tuple of:
            - DataFrame with model-specific evaluations merged
    """
    model_evaluations = load_model_evaluations_dict(variation)

    # Create list to hold each model's evaluation
    model_evals = []

    for model_combination in GLOBAL_APP_CONSTS.MODELS_COMBINATIONS:
        model_df = model_evaluations[model_combination]
        if prompt_idx not in model_df.index:
            continue

        row = model_df.loc[prompt_idx]

        # Filter out prompt-related columns (they're the same for all models)
        model_specific_data = {
            col: val for col, val in row.items() if col not in GLOBAL_APP_CONSTS.PROMPT_RELATED_COLUMNS
        }

        model_specific_data["model_arch"] = model_combination.arch
        model_specific_data["model_size"] = model_combination.size

        model_evals.append(model_specific_data)

    return pd.DataFrame(model_evals)


@CacheWithDependencies()
def get_models_remaining_prompts(
    model_combinations: list[MODEL_ARCH_AND_SIZE],
    window_size: TWindowSize,
    variation: TVariationName,
    prompt_original_indices: list[TPromptOriginalIndex],
) -> dict[MODEL_ARCH_AND_SIZE, HeatmapConfig]:
    """Get the remaining prompts for each model."""
    res = {}
    for model_arch, model_size in model_combinations:
        config = HeatmapConfig(
            variation=variation,
            common_params=CommonParams(
                model_arch=model_arch,
                model_size=model_size,
            ),
            prompt_filteration=SelectivePromptFilteration(
                dataset_name=DATASETS.COUNTER_FACT,
                prompt_ids=prompt_original_indices,
            ),
            runner_params=HeatmapParams(
                window_size=window_size,
            ),
        )
        if config.get_remaining_prompt_original_indices():
            res[MODEL_ARCH_AND_SIZE(model_arch, model_size)] = config
    return res


@CacheWithDependencies()
def load_model_combinations_prompts(
    variation: TVariationName, model_arch_and_sizes: list[MODEL_ARCH_AND_SIZE]
) -> list[ModelCombination]:
    """Get all possible model combinations and their corresponding prompts."""
    return get_model_combinations_prompts(variation, model_arch_and_sizes)
