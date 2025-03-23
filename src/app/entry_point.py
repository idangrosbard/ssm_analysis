from typing import assert_never

import streamlit as st

from src.app.app_consts import PAGE_ORDER
from src.app.page import (
    p01_home,
    p02_results_bank,
    p03_data_requirements,
    p04_heatmap_creation,
    p05_info_flow_analysis,
    p06_final_plots,
)
from src.utils.streamlit.helpers.component import StreamlitPage
from src.utils.streamlit.helpers.session_keys import mark_finished_global_refresh

st.set_page_config(layout="wide")


def get_page(page_order: PAGE_ORDER) -> StreamlitPage:
    match page_order:
        case PAGE_ORDER.HOME:
            return p01_home.HomePage()
        case PAGE_ORDER.RESULTS_BANK:
            return p02_results_bank.ResultsBankPage()
        case PAGE_ORDER.DATA_REQUIREMENTS:
            return p03_data_requirements.DataRequirementsPage()
        case PAGE_ORDER.HEATMAP:
            return p04_heatmap_creation.HeatmapCreationPage()
        case PAGE_ORDER.INFO_FLOW_ANALYSIS:
            return p05_info_flow_analysis.InfoFlowAnalysisPage()
        case PAGE_ORDER.FINAL_PLOTS:
            return p06_final_plots.FinalPlotsPage()
        case _:
            assert_never(page_order)


pg = st.navigation(
    {
        "Pages": [
            st.Page(page=get_page(page).render, title=page.title, icon=page.icon, url_path=page.name.lower())
            for page in PAGE_ORDER
        ]
    },
)

pg.run()

mark_finished_global_refresh()

# ###

# import shutil
# import time
# from pathlib import Path

# import pytest
# from datasets import DatasetDict
# from wfork_streamlit_profiler import Profiler

# from src.core.consts import PATHS, PathsConfig
# from src.core.names import COLS
# from src.core.types import (
#     ALL_SPLITS_LITERAL,
#     MODEL_ARCH,
#     SPLIT,
#     FeatureCategory,
#     TBatchSize,
#     TModelSize,
#     TokenType,
#     TPromptOriginalIndex,
#     TVariationName,
#     TWindowSize,
# )
# from src.data_ingestion.datasets.download_dataset import DATASETS
# from src.experiments.infrastructure.base_config import AllPromptFilteration, CommonParams, SelectivePromptFilteration
# from src.experiments.runners.full_pipeline import FullPipelineConfig, FullPipelineParam, main_local
# from src.experiments.runners.heatmap import HeatmapParams
# from src.experiments.runners.info_flow import InfoFlowParams, forward_eval
# from tests.src.experiments.test_full_pipeline import ORIGINAL_IDS, TEST_BASE_PATH

# with Profiler():
#     try:
#         prev_path = PATHS.PROJECT_DIR
#         PATHS.PROJECT_DIR = TEST_BASE_PATH
#         pipeline_config = FullPipelineConfig(
#             variation=TVariationName("test_baseline"),
#             runner_params=FullPipelineParam(
#                 knockout_map={
#                     TokenType.last: [
#                         (TokenType.last, FeatureCategory.ALL),
#                         (TokenType.subject, FeatureCategory.SLOW_DECAY),
#                         (TokenType.subject, FeatureCategory.FAST_DECAY),
#                         (TokenType.first, FeatureCategory.ALL),
#                         (TokenType.subject, FeatureCategory.ALL),
#                         (TokenType.relation, FeatureCategory.ALL),
#                     ],
#                     TokenType.subject: [
#                         (TokenType.context, FeatureCategory.ALL),
#                         (TokenType.subject, FeatureCategory.ALL),
#                     ],
#                     TokenType.relation: [
#                         (TokenType.context, FeatureCategory.ALL),
#                         (TokenType.subject, FeatureCategory.ALL),
#                         (TokenType.relation, FeatureCategory.ALL),
#                     ],
#                 },
#                 info_flow_window_size=TWindowSize(15),
#                 heatmap_window_size=TWindowSize(15),
#                 heatmap_prompts=SelectivePromptFilteration(
#                     DATASETS.COUNTER_FACT, cast(list[TPromptOriginalIndex], ORIGINAL_IDS[SPLIT.TRAIN1])
#                 ),
#                 with_plotting=True,
#                 enforce_no_missing_outputs=True,
#                 with_generation=True,
#             ),
#             common_params=CommonParams(
#                 model_arch=MODEL_ARCH.MAMBA1,
#                 model_size=TModelSize("130M"),
#             ),
#             prompt_filteration=ModelCorrectPromptFilteration(
#                 DATASETS.COUNTER_FACT,
#                 model_arch=MODEL_ARCH.MAMBA1,
#                 model_size=TModelSize("130M"),
#                 correctness=Correctness.correct,
#                 variation=TVariationName("test_baseline"),
#             ),
#         )

#         # uncomputed_dependencies = pipeline_config.uncomputed_dependencies()
#         if st.button("Compute"):
#             pipeline_config.compute_with_dependencies()

#     finally:
#         PATHS.PROJECT_DIR = prev_path
