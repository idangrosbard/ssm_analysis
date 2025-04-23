# Purpose: Provide components for generating plots based on plot plans
# High Level Outline:
# 1. Plot generation components for different plot types
# 2. Utility functions for loading and processing data
# 3. Plot rendering and saving functionality
# Outline Issues:
# - Consider adding more customization options for plots
# - Add support for interactive plots
# Outline Compatibility Issues:
# - New file, outline will be implemented

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import streamlit_antd_components as sac

from src.analysis.experiment_results.helpers import get_model_evaluations
from src.analysis.experiment_results.plot_plan import Cell, PlotPlan, get_hyper_param_definition
from src.analysis.plots.heatmaps import simple_diff_fixed
from src.analysis.plots.image_combiner import ImageGridParams, combine_image_grid
from src.analysis.plots.info_flow_confidence import create_confidence_plot
from src.app.texts import FINAL_PLOTS_TEXTS
from src.core.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID, TOKEN_TYPE_COLORS, TOKEN_TYPE_LINE_STYLES
from src.core.names import ExperimentName, SummarizedDataFulfilledReqsCols
from src.core.types import MODEL_ARCH_AND_SIZE, TPromptData
from src.data_ingestion.data_defs.data_defs import DataReqs, FulfilledReqs, PlotPlans, ResultBank
from src.data_ingestion.helpers.logits_utils import decode_tokens, get_prompt_row_index
from src.experiments.infrastructure.base_runner import InputParams
from src.experiments.infrastructure.setup_models import get_tokenizer
from src.experiments.runners.heatmap import HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowRunner
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.types_utils import class_values


@dataclass
class GridLayout:
    """Handles the organization and rendering of plots in a grid layout."""

    plot_plan: PlotPlan
    cells: list[Cell]
    data_reqs_per_cell: dict[Cell, DataReqs]
    plot_generator: "PlotGenerator"

    @property
    def row_values(self) -> list[Any]:
        """Get unique sorted row values."""
        return sorted({cell.rows for cell in self.cells})

    @property
    def col_values(self) -> list[Any]:
        """Get unique sorted column values."""
        return sorted({cell.cols for cell in self.cells})

    def get_cell_at(self, row_value: Any, col_value: Any) -> Optional[Cell]:
        """Get cell at the specified position."""
        return next((cell for cell in self.cells if cell.rows == row_value and cell.cols == col_value), None)

    def get_labels(self) -> tuple[list[str], list[str]]:
        """Get row and column labels."""
        row_labels = []
        col_labels = []

        # Get row labels
        for row_value in self.row_values:
            if row_value is not None:
                row_cell = next(cell for cell in self.cells if cell.rows == row_value)
                row_labels.append(row_cell.get_display_name("rows", self.plot_plan))

        # Get column labels
        for col_value in self.col_values:
            if col_value is not None:
                col_cell = next(cell for cell in self.cells if cell.cols == col_value)
                col_labels.append(col_cell.get_display_name("cols", self.plot_plan))

        return row_labels, col_labels

    def render_combined(self, recreate_plots: bool = False) -> None:
        """Render all plots combined into a single image."""
        image_grid: list[list[Optional[Path]]] = []
        row_labels, col_labels = self.get_labels()

        # Generate all plots and collect their paths
        for row_value in self.row_values:
            row_images: list[Optional[Path]] = []
            for col_value in self.col_values:
                cell = self.get_cell_at(row_value, col_value)
                if cell:
                    img_path = self.plot_generator._plot_cell(
                        self.data_reqs_per_cell[cell], cell, recreate_plots, show_plot=False
                    )
                    row_images.append(img_path)
                else:
                    row_images.append(None)
            image_grid.append(row_images)

        # Create grid params
        grid_params = ImageGridParams(
            row_labels=row_labels if row_labels else None,
            col_labels=col_labels if col_labels else None,
            img_width=800,  # Default width for plots
            img_height=600,  # Default height for plots
        )

        # Filter out None values from image grid
        filtered_grid = [[path for path in row if path is not None] for row in image_grid]
        filtered_grid = [row for row in filtered_grid if row]  # Remove empty rows

        # Combine images into a grid
        if filtered_grid:
            combined_image = combine_image_grid(filtered_grid, grid_params)
            if combined_image:
                st.image(combined_image)

    def render_separate(self, recreate_plots: bool = False) -> None:
        """Render plots in separate Streamlit columns."""
        has_row_labels = any(cell.rows is not None for cell in self.cells)
        has_col_labels = any(cell.cols is not None for cell in self.cells)
        row_labels, col_labels = self.get_labels()

        columns_count = ([0.5] if has_row_labels else []) + [1] * len(col_labels)

        # Create rows
        for i, row_value in enumerate(self.row_values):
            st_cols = st.columns(columns_count)

            # Show column headers if needed
            if i == 0 and has_col_labels:
                for col_name, col_col in zip((["Row"] if has_row_labels else []) + col_labels, st_cols):
                    col_col.write(f"**{col_name}**")

            # Add row label if needed
            start_col = 0
            if has_row_labels:
                with st_cols[0]:
                    st.write(f"**{row_labels[i]}**")
                start_col = 1

            # Add plots
            for col_value, col_col in zip(self.col_values, st_cols[start_col:]):
                cell = self.get_cell_at(row_value, col_value)
                if cell:
                    with col_col:
                        self.plot_generator._plot_cell(self.data_reqs_per_cell[cell], cell, recreate_plots)

    def render(self, recreate_plots: bool = False, combine_plots: bool = False) -> None:
        """Render the grid layout."""
        if combine_plots:
            self.render_combined(recreate_plots)
        else:
            self.render_separate(recreate_plots)


class Tabs:
    PLOT_INDIVIDUAL = "Plot Individually"
    PLOT_COMBINED = "Plot Combined"
    CUSTOMIZE_PLOT = "Customize Plot"


class PlotGenerator(StreamlitComponent[Optional[str]]):
    """Component for generating plots based on plot plans."""

    def __init__(self, plot_plan: PlotPlan, result_bank: ResultBank):
        self.plot_plan = plot_plan
        self.result_bank = result_bank

    def _get_model_display_name(self, model_arch_and_size: MODEL_ARCH_AND_SIZE) -> str:
        return model_arch_and_size.model_name

    def _get_cell_cache_path(self, grid_name: Any, row_name: Any, col_name: Any) -> Path:
        """Generate a unique cache path for a cell's plot."""
        cache_dir = PlotPlans.get_cache_dir(self.plot_plan.plot_id)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique identifier for the cell
        cell_id = f"{grid_name}_{row_name}_{col_name}".replace(" ", "_")
        return cache_dir / f"{cell_id}.png"

    def _plot_cell(
        self,
        data_reqs: DataReqs,
        cell: Cell,
        recreate: bool = False,
        with_plotly: bool = False,
        show_plot: bool = True,
    ) -> Path:
        """Plot a single cell with caching."""
        cache_path = cell.get_cache_path(self.plot_plan, PlotPlans.get_cache_dir(self.plot_plan.plot_id))

        if not recreate and cache_path.exists():
            # Load and display cached plot if needed
            if show_plot:
                st.image(str(cache_path))
            return cache_path

        # Get fulfilled requirements
        fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank)

        # Create the plot based on plot type
        fig = None
        match self.plot_plan.experiment_name:
            case ExperimentName.info_flow:
                fig = self._generate_cell_knockout(fulfilled_reqs)
            case ExperimentName.heatmap:
                fulfilled_reqs = fulfilled_reqs.choose_latest_fulfilled()
                items = list(fulfilled_reqs.items())
                assert len(items) == 1
                filterations, runners = items[0][1]
                prompt_idx = filterations.get_prompt_ids()
                assert len(prompt_idx) == 1
                assert len(runners) == 1
                runner = runners[0].modify(input_params=InputParams(filteration=filterations))
                assert isinstance(runner, HeatmapRunner)
                prompt_id = prompt_idx[0]
                model_arch_and_size = MODEL_ARCH_AND_SIZE(
                    runner.variant_params.model_arch, runner.variant_params.model_size
                )
                data = cast(
                    TPromptData,
                    get_model_evaluations(runner.metadata_params.code_version, [model_arch_and_size])[
                        model_arch_and_size
                    ],
                )
                tokenizer = get_tokenizer(runner.variant_params.model_arch, runner.variant_params.model_size)
                model_id = MODEL_SIZES_PER_ARCH_TO_MODEL_ID[runner.variant_params.model_arch][
                    runner.variant_params.model_size
                ]
                prob_mat = runner.get_outputs()[prompt_id]
                prompt = get_prompt_row_index(data, prompt_id)
                input_ids = prompt.input_ids(tokenizer, "cpu")
                toks = cast(list[str], decode_tokens(tokenizer, input_ids[0]))
                last_tok = toks[-1]
                toks[-1] = toks[-1] + "*"

                fig, _ = simple_diff_fixed(
                    prob_mat=prob_mat,
                    model_id=model_id,
                    window_size=runner.variant_params.window_size,
                    last_tok=last_tok,
                    base_prob=prompt.base_prob,
                    true_word=prompt.true_word,
                    toks=toks,
                    fixed_diff=0.3,
                )

            case _:
                raise ValueError(f"Unknown experiment name: {self.plot_plan.experiment_name}")

        if fig is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(fig, go.Figure):
                fig.write_image(str(cache_path), scale=4)
            else:
                # Save the plot
                plt.savefig(str(cache_path), bbox_inches="tight")
                plt.close(fig)

            # Display the plot if needed
            if show_plot:
                if with_plotly:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.image(str(cache_path))

        return cache_path

    def _generate_cell_knockout(self, fulfilled_reqs: FulfilledReqs):
        """Generate knockout plot for a single cell."""
        # Create the base figure
        configs = list(fulfilled_reqs.get_config().values())
        data = []
        title = "-".join(
            [
                # config.common_params.model_arch,
                # config.common_params.model_size,
                # str(config.runner_params.window_size),
            ]
        )
        for config in configs:
            assert isinstance(config, InfoFlowRunner)

            try:
                data.append(
                    {
                        "label": f"{config.variant_params.source} - {config.variant_params.feature_category}",
                        "color": TOKEN_TYPE_COLORS.get(config.variant_params.source, "#000000"),
                        "linestyle": TOKEN_TYPE_LINE_STYLES.get(config.variant_params.feature_category, "-"),
                        "data": config.get_outputs(),
                    }
                )
            except Exception as e:
                # TODO: remove
                print(e)
                print(config.variant_params)
                config.output_file.get_statistics.cache_clear()  # type: ignore
                if config.output_file.statistics_path.exists():
                    config.output_file.statistics_path.unlink()
                data.append(
                    {
                        "label": f"{config.variant_params.source} - {config.variant_params.feature_category}",
                        "color": TOKEN_TYPE_COLORS.get(config.variant_params.source, "#000000"),
                        "linestyle": TOKEN_TYPE_LINE_STYLES.get(config.variant_params.feature_category, "-"),
                        "data": config.get_outputs(),
                    }
                )
        with_fixed_limits = False
        fig = create_confidence_plot(
            lines_metadata=data,
            confidence_level=0.95,
            title=title,
            plots_meta_data={
                "acc": {
                    "title": "Accuracy",
                    "ylabel": "% accuracy",
                    "ylabel_loc": "center",
                    "axhline_value": 100.0,
                    "ylim": (60.0, 105.0) if with_fixed_limits else None,
                },
                "diff": {
                    "title": "Normalized change in prediction probability",
                    "ylabel": "% probability change",
                    "ylabel_loc": "top",
                    "axhline_value": 0.0,
                    "ylim": (-50.0, 50.0) if with_fixed_limits else None,
                },
            },
        )

        return fig

    def render(self) -> Optional[str]:
        """Generate and display a plot based on the plot plan."""
        st.subheader(f"{FINAL_PLOTS_TEXTS.generating_plot(self.plot_plan.title)}")

        # Check if we have all the required data
        data_reqs = self.plot_plan.get_data_requirements(self.result_bank)
        summarized_fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank).summarize()
        reqs = summarized_fulfilled_reqs.to_data_reqs().to_rows()

        missing_reqs = [
            req
            for summary, req in zip(summarized_fulfilled_reqs, reqs)
            if summary[SummarizedDataFulfilledReqsCols.AvailableOptions] == 0
        ]

        if missing_reqs:
            st.error(f"Missing data for {len(missing_reqs)} requirements. Please run the missing requirements first.")
            return None

        # Generate the plot based on the plot type
        data_reqs_per_cell = self.plot_plan.get_data_requirements_per_cell(self.result_bank)

        # Add checkbox for plot recreation
        tab = sac.tabs([sac.TabsItem(label=tab_name) for tab_name in class_values(Tabs)])
        combine_plots = tab == Tabs.PLOT_COMBINED

        if tab == Tabs.CUSTOMIZE_PLOT:
            pass

        else:
            recreate_plots = tab == Tabs.PLOT_INDIVIDUAL and st.checkbox("Recreate all plots", value=False)

            # Group cells by grid
            cells_by_grid: dict[Any, list[Cell]] = {}
            for cell in data_reqs_per_cell:
                cells_by_grid.setdefault(cell.grids, []).append(cell)

            # Create tabs for different plot views
            if len(cells_by_grid) == 1 and None in cells_by_grid:
                tabs = [st.empty()]
                grid_names = [None]
            else:
                grid_options = self.plot_plan.grids
                assert grid_options is not None
                grid_param_definition = get_hyper_param_definition(grid_options)
                grid_names = sorted(cells_by_grid.keys())
                tabs = st.tabs([grid_param_definition.get_display_name(grid) for grid in grid_names])

            # Render each grid
            for grid_name, tab in zip(grid_names, tabs):
                with tab:
                    grid_layout = GridLayout(
                        plot_plan=self.plot_plan,
                        cells=cells_by_grid[grid_name],
                        data_reqs_per_cell=data_reqs_per_cell,
                        plot_generator=self,
                    )
                    grid_layout.render(recreate_plots, combine_plots)
