from pathlib import Path
from typing import NamedTuple, Optional, cast

import pandas as pd

from src.consts import (
    EXPERIMENT_NAMES,
    GRAPHS_ORDER,
    MODEL_ARCH,
    MODEL_SIZE_CAT,
    PATHS,
    TokenType,
    is_falcon,
    is_mamba_arch,
)
from src.final_plots.results_bank import HeatmapRecord, InfoFlowRecord, ResultRecord, get_results_bank
from src.types import FeatureCategory


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


IDataFulfilled = dict[DataReq, Optional[Path]]
IDataFulfilledOptions = dict[DataReq, list[Path]]

DATA_FULFILLED_OVERIDES_PATH = Path(__file__).parent / "data_fulfilled_overides.csv"
DATA_FULFILLED_PATH = Path(__file__).parent / "data_fulfilled.csv"


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
    results = get_results_bank()
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

    for model_arch, model_size, _ in GRAPHS_ORDER:
        for is_all_correct in [True, False]:
            for source in [TokenType.last, TokenType.first, TokenType.subject, TokenType.relation]:
                data_reqs[
                    DataReq(
                        experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                        model_arch=model_arch,
                        model_size=model_size,
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

    for model_arch, model_size, _ in GRAPHS_ORDER:
        if is_mamba_arch(model_arch):
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
                            model_arch=model_arch,
                            model_size=model_size,
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

    for model_arch, model_size, _ in GRAPHS_ORDER:
        if is_falcon(model_size):
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
                        model_arch=model_arch,
                        model_size=model_size,
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

    for model_arch, model_size, model_size_cat in GRAPHS_ORDER:
        if model_size_cat != MODEL_SIZE_CAT.LARGE:
            continue
        for is_all_correct in [True, False]:
            for source in [TokenType.context, TokenType.subject]:
                data_reqs[
                    DataReq(
                        experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                        model_arch=model_arch,
                        model_size=model_size,
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

    for model_arch, model_size, _ in GRAPHS_ORDER:
        if is_mamba_arch(model_arch):
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
                            model_arch=model_arch,
                            model_size=model_size,
                            window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                            is_all_correct=is_all_correct,
                            source=source,
                            feature_category=feature_category,
                            target=TokenType.last,
                            prompt_idx=None,
                        )
                    ] = None

    # endregion

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
    for model_arch, model_size, _ in GRAPHS_ORDER:
        if is_mamba_arch(model_arch):
            for window_size in ALL_WINDOW_SIZES:
                for source in [TokenType.last, TokenType.first, TokenType.subject, TokenType.relation]:
                    data_reqs[
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                            model_arch=model_arch,
                            model_size=model_size,
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

    for model_arch, model_size, _ in GRAPHS_ORDER:
        if is_mamba_arch(model_arch):
            for is_all_correct in [True, False]:
                for source in [TokenType.last, TokenType.first, TokenType.subject, TokenType.relation]:
                    data_reqs[
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                            model_arch=model_arch,
                            model_size=model_size,
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


def update_data_reqs_with_latest_results() -> None:
    data_reqs_options = get_latest_data_fulfilled()
    loaded_overides: dict[DataReq, Path | None] = load_data_fulfilled_overides()
    _save_data_fulfilled(DATA_FULFILLED_PATH, merge_data_reqs(loaded_overides, data_reqs_options, keys_by_first=False))
