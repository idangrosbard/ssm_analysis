from src.analysis.experiment_results.data_requirements import (
    DataReq,
    IDataFulfilled,
)
from src.core.consts import (
    GRAPHS_ORDER,
    MODEL_SIZE_CAT,
    TokenType,
    is_falcon,
    is_mamba_arch,
)
from src.core.names import EXPERIMENT_NAMES
from src.core.types import (
    FeatureCategory,
    TWindowSize,
)
from src.data_ingestion.data_defs import DataReqs

STANDARD_WINDOW_SIZE_FOR_INFO_FLOW = TWindowSize(9)
STANDARD_WINDOW_SIZE_FOR_HEATMAP = TWindowSize(5)
ALL_WINDOW_SIZES = [TWindowSize(size) for size in [1, 3, 5, 9, 12, 15]]


def get_default_data_reqs() -> DataReqs:
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
    feature_category = [FeatureCategory.ALL]
    target = [last]
    source = [last, first, subject, relation]
    """

    for model_arch_and_size in GRAPHS_ORDER:
        for source in [
            TokenType.last,
            TokenType.first,
            TokenType.subject,
            TokenType.relation,
        ]:
            data_reqs[
                DataReq(
                    experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                    model_arch=model_arch_and_size.arch,
                    model_size=model_arch_and_size.size,
                    window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                    source=source,
                    feature_category=FeatureCategory.ALL,
                    target=TokenType.last,
                    prompt_idx=None,
                ).validate()
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
    target = [last]
    source = [last, first, subject, relation, subject-SLOW_DECAY, subject-FAST_DECAY]
    """

    for model_arch_and_size in GRAPHS_ORDER:
        if is_mamba_arch(model_arch_and_size.arch):
            for source, feature_category in [
                (TokenType.last, FeatureCategory.ALL),
                (TokenType.first, FeatureCategory.ALL),
                (TokenType.subject, FeatureCategory.ALL),
                (TokenType.relation, FeatureCategory.ALL),
                (TokenType.subject, FeatureCategory.SLOW_DECAY),
                (TokenType.subject, FeatureCategory.FAST_DECAY),
            ]:
                data_reqs[
                    DataReq(
                        experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                        model_arch=model_arch_and_size.arch,
                        model_size=model_arch_and_size.size,
                        window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                        source=source,
                        feature_category=feature_category,
                        target=TokenType.last,
                        prompt_idx=None,
                    ).validate()
                ] = None

    # endregion

    # region 3. Figure 3 Knockout information flow to the last token - Falcon Mamba 7B.
    """
    Figure 3: Knockout information flow to the last token - Falcon Mamba 7B:
    model sizes = [HUGE]
    model archs = [Falcon Mamba]
    window sizes = [STANDARD_WINDOW_SIZE_FOR_INFO_FLOW]
    target = [last]
    source = [last, first, subject, relation, subject-SLOW_DECAY, subject-FAST_DECAY]
    """

    for model_arch_and_size in GRAPHS_ORDER:
        if is_falcon(model_arch_and_size.size):
            for source, feature_category in [
                (TokenType.last, FeatureCategory.ALL),
                (TokenType.first, FeatureCategory.ALL),
                (TokenType.subject, FeatureCategory.ALL),
                (TokenType.relation, FeatureCategory.ALL),
                (TokenType.subject, FeatureCategory.SLOW_DECAY),
                (TokenType.subject, FeatureCategory.FAST_DECAY),
            ]:
                data_reqs[
                    DataReq(
                        experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                        model_arch=model_arch_and_size.arch,
                        model_size=model_arch_and_size.size,
                        window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                        source=source,
                        feature_category=feature_category,
                        target=TokenType.last,
                        prompt_idx=None,
                    ).validate()
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
    target = [subject]
    source = [context, subject]
    """

    for model_arch_and_size, model_size_cat in GRAPHS_ORDER.items():
        if model_size_cat != MODEL_SIZE_CAT.LARGE:
            continue
        for source in [TokenType.context, TokenType.subject]:
            data_reqs[
                DataReq(
                    experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                    model_arch=model_arch_and_size.arch,
                    model_size=model_arch_and_size.size,
                    window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                    source=source,
                    feature_category=FeatureCategory.ALL,
                    target=TokenType.subject,
                    prompt_idx=None,
                ).validate()
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
    target = [last]
    source = [context, subject, relation, subject-SLOW_DECAY, subject-FAST_DECAY]
    """

    for model_arch_and_size in GRAPHS_ORDER:
        if is_mamba_arch(model_arch_and_size.arch):
            for source, feature_category in [
                (TokenType.last, FeatureCategory.ALL),
                (TokenType.subject, FeatureCategory.SLOW_DECAY),
                (TokenType.subject, FeatureCategory.FAST_DECAY),
                (TokenType.first, FeatureCategory.ALL),
                (TokenType.subject, FeatureCategory.ALL),
                (TokenType.relation, FeatureCategory.ALL),
            ]:
                data_reqs[
                    DataReq(
                        experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                        model_arch=model_arch_and_size.arch,
                        model_size=model_arch_and_size.size,
                        window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                        source=source,
                        feature_category=feature_category,
                        target=TokenType.last,
                        prompt_idx=None,
                    ).validate()
                ] = None

    # endregion

    # also for 'last' target with all vars of decay only for mamba1 & 2 of the large sizes

    for source in [TokenType.relation, TokenType.first, TokenType.last, TokenType.all]:
        for model_arch_and_size, model_size_cat in GRAPHS_ORDER.items():
            if is_mamba_arch(model_arch_and_size.arch) and model_size_cat == MODEL_SIZE_CAT.LARGE:
                for feature_category in [
                    FeatureCategory.SLOW_DECAY,
                    FeatureCategory.FAST_DECAY,
                ]:
                    data_reqs[
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                            model_arch=model_arch_and_size.arch,
                            model_size=model_arch_and_size.size,
                            window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                            source=source,
                            feature_category=feature_category,
                            target=TokenType.last,
                            prompt_idx=None,
                        ).validate()
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
    target = [last]
    source = [last, first, subject, relation]

    """
    for model_arch_and_size in GRAPHS_ORDER:
        if is_mamba_arch(model_arch_and_size.arch):
            for window_size in ALL_WINDOW_SIZES:
                for source in [
                    TokenType.last,
                    TokenType.first,
                    TokenType.subject,
                    TokenType.relation,
                ]:
                    data_reqs[
                        DataReq(
                            experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                            model_arch=model_arch_and_size.arch,
                            model_size=model_arch_and_size.size,
                            window_size=window_size,
                            source=source,
                            feature_category=FeatureCategory.ALL,
                            target=TokenType.last,
                            prompt_idx=None,
                        ).validate()
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
    target = [last]
    source = [last, first, subject, relation]
    """

    for model_arch_and_size in GRAPHS_ORDER:
        if is_mamba_arch(model_arch_and_size.arch):
            for source in [
                TokenType.last,
                TokenType.first,
                TokenType.subject,
                TokenType.relation,
            ]:
                data_reqs[
                    DataReq(
                        experiment_name=EXPERIMENT_NAMES.INFO_FLOW,
                        model_arch=model_arch_and_size.arch,
                        model_size=model_arch_and_size.size,
                        window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                        source=source,
                        feature_category=FeatureCategory.ALL,
                        target=TokenType.last,
                        prompt_idx=None,
                    ).validate()
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

    return DataReqs(set(data_reqs.keys()))
