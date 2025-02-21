from pathlib import Path
from typing import Any, Callable, Optional, TypedDict, TypeVar, cast

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from streamlit.elements.arrow import DataframeState

from src.consts import PATHS
from src.experiments.heatmap import HeatmapConfig
from src.final_plots.app.app_consts import HeatmapCols, HeatmapConsts
from src.final_plots.data_reqs import DataReq, get_current_data_reqs
from src.final_plots.results_bank import ParamNames, clear_results_bank_cache
from src.types import MODEL_ARCH

T = TypeVar("T")


class PaginationConfig(TypedDict):
    page_size: int
    current_page: int


def create_pagination_config(
    total_items: int,
    default_page_size: int = 10,
    pages_options: tuple[int, ...] = (5, 10, 20, 50, 100),
    key_prefix: str = "",
    on_change: Optional[Callable[[], None]] = None,
) -> PaginationConfig:
    """Create pagination configuration.

    Args:
        total_items: Total number of items to paginate
        default_page_size: Default number of items per page
        key_prefix: Prefix for session state keys to avoid conflicts

    Returns:
        PaginationConfig with page size and current page
    """
    # Create columns for pagination controls
    cols = st.columns(
        [10, 1],
        vertical_alignment="bottom",
    )
    with cols[1]:
        page_size = st.selectbox(
            "Items per page",
            options=pages_options,
            index=pages_options.index(st.session_state[f"{key_prefix}page_size"]),
        )
    with cols[0]:
        current_page = sac.pagination(
            total=total_items,
            page_size=page_size,
            align="center",
            show_total=True,
            jump=True,
            variant="filled",
            # size=50,
            on_change=cast(Callable[[], Any], on_change),
        )

    assert isinstance(current_page, int)
    return {
        "page_size": page_size,
        "current_page": current_page,
    }


def apply_pagination(df: pd.DataFrame, pagination_config: PaginationConfig) -> pd.DataFrame:
    """Apply pagination to a DataFrame.

    Args:
        df: DataFrame to paginate
        pagination_config: Pagination configuration

    Returns:
        Paginated DataFrame
    """
    start_idx = pagination_config["current_page"] * pagination_config["page_size"]
    end_idx = start_idx + pagination_config["page_size"]
    return df.iloc[start_idx:end_idx]


def create_filters(
    df: pd.DataFrame,
    filter_columns: list[str],
    exclude_columns: list[str] = [],
    default_values: dict[str, list] = {},
) -> dict[str, list]:
    """Create sidebar filters for the given dataframe columns.

    Args:
        df: DataFrame to create filters for
        filter_columns: List of columns to create filters for
        exclude_columns: List of columns to exclude from filtering
        default_values: Dictionary of default values for filters

    Returns:
        Dictionary of selected filter values for each column
    """

    with st.sidebar:
        with st.expander("Filters"):
            filters = {}

            # Create filters for each column
            for col in filter_columns:
                if col in exclude_columns:
                    continue

                unique_values = sorted(df[col].dropna().unique())
                if len(unique_values) <= 1:
                    continue

                filters[col] = st.multiselect(f"Filter {col}", unique_values, default=default_values.get(col, []))

            return filters


def apply_filters(df: pd.DataFrame, filters: dict[str, list]) -> pd.DataFrame:
    """Apply filters to the dataframe.

    Args:
        df: DataFrame to filter
        filters: Dictionary of filter values for each column

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    for col, selected_values in filters.items():
        if selected_values:
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    return filtered_df


def get_data_req_from_df_row(row: pd.Series) -> DataReq:
    return DataReq(
        **{param: row[param] for param in ParamNames if param not in [ParamNames.path, ParamNames.variation]},
    )


def my_cache_data(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to cache data with a refresh button in the sidebar.

    Args:
        func: Function to cache

    Returns:
        Cached function with refresh button
    """
    cached_func: Callable[[], T] = st.cache_data(func)  # type: ignore

    def wrapper(*args: Any, **kwargs: Any) -> T:
        if st.sidebar.button("🔄 Refresh Data"):
            # Clear all caches
            clear_results_bank_cache()
            cached_func.clear()  # type: ignore
            st.rerun()
        return cached_func(*args, **kwargs)

    return wrapper


def format_path_for_display(path: Path | str | None) -> str:
    """Format a path for display in the UI.

    Args:
        path: Path to format

    Returns:
        Formatted path string
    """
    if path is None:
        return ""

    if isinstance(path, Path):
        try:
            return str(path.relative_to(PATHS.PROJECT_DIR))
        except ValueError:
            return str(path)

    return format_path_for_display(Path(path))


def show_filtered_count(filtered_df: pd.DataFrame, total_df: pd.DataFrame, item_name: str = "results") -> None:
    """Show the count of filtered items vs total items.

    Args:
        filtered_df: Filtered DataFrame
        total_df: Total DataFrame
        item_name: Name of the items being counted
    """
    st.write(f"Showing {len(filtered_df)} {item_name} out of {len(total_df)} total")


def load_experiment_data(experiment_name: str) -> pd.DataFrame:
    """Load data for a specific experiment.

    Args:
        experiment_name: Name of the experiment to load data for

    Returns:
        DataFrame with experiment data
    """
    current_fulfilled = get_current_data_reqs()

    data = []
    for req, data_path in current_fulfilled.items():
        if not hasattr(req, "experiment_name") or req.experiment_name.value != experiment_name:
            continue

        if not data_path:  # Skip requirements with no fulfillment
            continue

        row_dict = {
            param: getattr(req, param, None)
            for param in ParamNames
            if param not in [ParamNames.path, ParamNames.variation]
        }
        row = {**row_dict, "data_path": data_path}
        data.append(row)

    return pd.DataFrame(data)


def get_param_values(df: pd.DataFrame, param: str) -> list[Any]:
    """Get unique values for a parameter from the dataframe.

    Args:
        df: DataFrame to get values from
        param: Parameter to get values for

    Returns:
        List of unique values
    """
    return sorted(df[param].unique())


def filter_combinations(df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    df = df.copy()

    modification_container = st.container()

    model_name_filters = []
    for model_name in model_names:
        model_name_filters.append((model_name, True))
        model_name_filters.append((f"{model_name}", False))

    def format_func(option: str | tuple[str, bool]) -> str:
        if isinstance(option, tuple):
            return f"{option[0]} - {'correct ✅' if option[1] else 'incorrect ❌'}"
        else:
            assert option == HeatmapCols.PROMPT_COUNT
            return "Prompt Count"

    with modification_container:
        to_filter_columns = st.multiselect(
            "Filter dataframe on",
            [HeatmapCols.PROMPT_COUNT, *model_name_filters],
            format_func=format_func,
            default=[HeatmapCols.PROMPT_COUNT],
        )

        # Existing column filtering logic
        for column in to_filter_columns:
            if column == HeatmapCols.PROMPT_COUNT:
                _min = int(df[column].min())
                _max = int(df[column].max())
                user_num_input = st.number_input(
                    "Minimum prompt count",
                    min_value=_min,
                    max_value=_max,
                    value=HeatmapConsts.MINIMUM_COMBINATIONS_FOR_FILTERING,
                    step=1,
                )
                df = df[df[column] >= user_num_input]
            else:
                assert isinstance(column, tuple)
                model_name, is_correct = column
                if is_correct:
                    df = df[df[model_name] == "✅"]
                else:
                    df = df[df[model_name] == "❌"]

        # New minimum correct models filter
        min_correct = st.number_input(
            "Minimum correct models",
            min_value=0,
            max_value=len(model_names),
            value=0,
            help="Show only combinations with at least this many correct models",
        )
        if min_correct > 0:
            # Count number of ✅ in model columns for each row
            df = df[df[model_names].apply(lambda row: (row == "✅").sum(), axis=1) >= min_correct]

    return df


def get_model_heatmap_config(
    model_arch: MODEL_ARCH, model_size: str, window_size: int, variation: str, prompt_original_indices: list[int]
) -> HeatmapConfig:
    """Get the data requirement for a specific model.

    Args:
        model_arch: Model architecture
        model_size: Model size

    Returns:
        Data requirement for the model
    """
    return HeatmapConfig(
        model_arch=model_arch,
        model_size=model_size,
        window_size=window_size,
        variation=variation,
        prompt_original_indices=prompt_original_indices,
    )


def get_steamlit_dataframe_selected_row(selected_row: Optional[DataframeState]) -> Optional[int]:
    if selected_row is not None:
        assert "selection" in selected_row
        assert "rows" in selected_row["selection"]
        if len(selected_row["selection"]["rows"]) == 1:
            return selected_row["selection"]["rows"][0]
