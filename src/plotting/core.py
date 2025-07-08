from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, cast

import matplotlib.figure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from more_itertools import unique_everseen
from PIL.Image import Image
from pydantic import BaseModel

from src.analysis.experiment_results.helpers import get_model_evaluations
from src.analysis.experiment_results.hyper_param_definition import FilterationFactoryHPD
from src.analysis.experiment_results.plot_plan import Cell, PlotPlan
from src.analysis.experiment_results.prompt_filteration_factory import (
    PromptFilterationFactory,
)
from src.analysis.plots.heatmaps import HeatmapPlotConfig, simple_diff_fixed
from src.analysis.plots.image_combiner import (
    ImageGridParams,
    LegendItem,
    combine_image_grid,
)
from src.analysis.plots.info_flow_confidence import (
    InfoFlowPlotConfig,
    create_confidence_plot,
)
from src.core.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID
from src.core.names import ExperimentName, FinalPlotsPlanOrientation
from src.core.types import (
    MODEL_ARCH_AND_SIZE,
    TInfoFlowOutput,
    TLineStyle,
    TPromptData,
)
from src.data_ingestion.data_defs.data_defs import (
    DataReqs,
    PlotPlans,
    ResultBank,
)
from src.data_ingestion.helpers.logits_utils import (
    decode_tokens,
    get_prompt_row_index,
)
from src.experiments.infrastructure.base_runner import BaseRunner
from src.experiments.infrastructure.setup_models import get_tokenizer
from src.experiments.runners.heatmap import HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowRunner
from src.utils.infra.image_utils import save_at_dpi


def get_runners(data_reqs: DataReqs, result_bank: ResultBank) -> list[BaseRunner]:
    return list(data_reqs.to_fulfilled_reqs(result_bank).choose_latest_fulfilled().get_config().values())


@dataclass
class PlotGenerator:
    """Component for generating plots based on plot plans."""

    plot_plan: PlotPlan
    result_bank: ResultBank

    def _get_legend_items(self, relevant_data_reqs_list: Optional[list[DataReqs]] = None) -> list[LegendItem]:
        plot_config = self._get_config_for_experiment_name(self.plot_plan.experiment_name, None)
        legend_items = []
        if isinstance(plot_config, InfoFlowPlotConfig):
            relevant_line_ids: Optional[set[str]] = None
            if relevant_data_reqs_list:
                relevant_line_ids = set()
                lines_param_config = self.plot_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation.lines)
                assert lines_param_config is not None
                original_lines_hpd = lines_param_config.get_param_def()

                for data_reqs in relevant_data_reqs_list:
                    runners = get_runners(data_reqs, self.result_bank)
                    for runner_instance in runners:
                        if isinstance(runner_instance, InfoFlowRunner):
                            line_id_str = original_lines_hpd.get_line_id_from_runner(runner_instance.variant_params)
                            relevant_line_ids.add(line_id_str)

            for line_id, color in plot_config.custom_colors.items():
                if relevant_line_ids is None or line_id in relevant_line_ids:
                    legend_items.append(
                        LegendItem(
                            label=plot_config.custom_line_labels.get(line_id, line_id),
                            color=color.as_hex(),
                            linestyle=plot_config.custom_line_styles.get(line_id, TLineStyle.solid.value),
                        )
                    )
        return legend_items

    def plot_data_reqs(
        self, data_reqs: DataReqs, cell_plot_config: dict[str, Any]
    ) -> matplotlib.figure.Figure | go.Figure | None:
        runners = get_runners(data_reqs, self.result_bank)

        fig = None
        match self.plot_plan.experiment_name:
            case ExperimentName.info_flow:
                config = self._get_config_for_experiment_name(self.plot_plan.experiment_name, cell_plot_config)
                assert isinstance(config, InfoFlowPlotConfig)
                fig = self._generate_cell_knockout(runners, config)
            case ExperimentName.heatmap:
                assert len(runners) == 1
                runner = runners[0]
                assert isinstance(runner, HeatmapRunner)
                prompt_idx = runner.input_params.filteration.get_prompt_ids()
                assert len(prompt_idx) == 1
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
                for i, tok in enumerate(toks):
                    if input_ids[0][i] == tokenizer.bos_token_id:
                        toks[i] = "<BOS>"
                    if input_ids[0][i] == tokenizer.eos_token_id:
                        toks[i] = "<EOS>"
                last_tok = toks[-1]
                toks[-1] = toks[-1] + "*"

                config = self._get_config_for_experiment_name(self.plot_plan.experiment_name, cell_plot_config)
                assert isinstance(config, HeatmapPlotConfig)

                fig, _ = simple_diff_fixed(
                    prob_mat=prob_mat,
                    model_id=model_id,
                    window_size=runner.variant_params.window_size,
                    last_tok=last_tok,
                    base_prob=prompt.base_prob,
                    target_rank=prompt.target_rank,
                    true_word=prompt.true_word,
                    toks=toks,
                    config=config,
                )

            case _:
                raise ValueError(f"Unknown experiment name: {self.plot_plan.experiment_name}")

        return fig

    def plot_cell_to_path(
        self,
        data_reqs: DataReqs,
        cell: Cell,
        recreate: bool = False,
    ) -> Path:
        """Plot a single cell with caching."""
        cache_path = PlotPlans.get_cell_cache_path(self.plot_plan, cell)
        if not recreate and cache_path.exists():
            return cache_path

        cell_plot_config = self.plot_plan.cell_plot_config
        if isinstance(cell_plot_config, BaseModel):
            cell_plot_config = cell_plot_config.model_dump()
        # If already a dict, use as-is
        fig = self.plot_data_reqs(data_reqs, cell_plot_config)
        if fig is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(fig, go.Figure):
                fig.write_image(str(cache_path), scale=4)
            elif isinstance(fig, matplotlib.figure.Figure):
                fig.savefig(str(cache_path), dpi=600)
                plt.close(fig)
            else:
                raise TypeError(f"Plot for {cache_path.name} was of unexpected type {type(fig)}")

        return cache_path

    def _generate_cell_knockout(self, runners: list[BaseRunner], cell_plot_config: InfoFlowPlotConfig):
        """Generate knockout plot for a single cell."""
        data: dict[str, TInfoFlowOutput] = {}

        lines_param_config = self.plot_plan.get_param_config_by_orientation(FinalPlotsPlanOrientation.lines)
        assert lines_param_config is not None
        original_lines_hpd = lines_param_config.get_param_def()

        for i, runner_instance in enumerate(runners):
            assert isinstance(runner_instance, InfoFlowRunner), f"Expected InfoFlowRunner, got {type(runner_instance)}"

            if isinstance(original_lines_hpd, FilterationFactoryHPD):
                cur_prompt_filteration = lines_param_config.values[i]
                assert isinstance(cur_prompt_filteration, PromptFilterationFactory)
                line_id_str = original_lines_hpd.get_display_name(cur_prompt_filteration)
            else:
                line_id_str = original_lines_hpd.get_line_id_from_runner(runner_instance.variant_params)
            data[line_id_str] = runner_instance.get_outputs()

        fig = create_confidence_plot(
            lines=data,
            confidence_level=cell_plot_config.confidence_level,
            config=cell_plot_config,
        )

        return fig

    def _get_model_for_experiment_name(self, experiment_name: ExperimentName):
        """Get the appropriate configuration model based on experiment name."""
        if experiment_name == ExperimentName.info_flow:
            return InfoFlowPlotConfig.specify_config(
                self.plot_plan.get_option_display_names_for_orientation(FinalPlotsPlanOrientation.lines)
            )
        elif experiment_name == ExperimentName.heatmap:
            return HeatmapPlotConfig
        else:
            raise ValueError(f"Experiment name {experiment_name} is not implemented")

    def _get_config_for_experiment_name(
        self,
        experiment_name: ExperimentName,
        config: Optional[BaseModel | dict[str, Any]],
    ):
        """Get the appropriate configuration model based on experiment name."""
        if config is None:
            config = self.plot_plan.cell_plot_config
        if isinstance(config, BaseModel):
            config = config.model_dump()  # type: ignore[reportOptionalMemberAccess]
        # If config is already a dict, use as-is
        return self._get_model_for_experiment_name(experiment_name).model_validate(config)


@dataclass
class GridLayout:
    """Handles the organization of plots in a grid layout."""

    plot_plan: PlotPlan
    cells: list[Cell]
    data_reqs_per_cell: dict[Cell, DataReqs]
    plot_generator: "PlotGenerator"

    @property
    def row_values(self) -> Iterator[Any]:
        return unique_everseen([cell.rows for cell in self.cells])

    @property
    def col_values(self) -> Iterator[Any]:
        return unique_everseen([cell.cols for cell in self.cells])

    def get_cell_at(self, row_value: Any, col_value: Any) -> Optional[Cell]:
        """Get cell at the specified position."""
        return next(
            (cell for cell in self.cells if cell.rows == row_value and cell.cols == col_value),
            None,
        )

    def get_labels(self) -> tuple[list[str], list[str]]:
        """Get row and column labels."""
        row_labels = []
        col_labels = []

        for row_value in self.row_values:
            if row_value is not None:
                row_cell = next(cell for cell in self.cells if cell.rows == row_value)
                row_labels.append(row_cell.get_field_display_name("rows", self.plot_plan))

        for col_value in self.col_values:
            if col_value is not None:
                col_cell = next(cell for cell in self.cells if cell.cols == col_value)
                col_labels.append(col_cell.get_field_display_name("cols", self.plot_plan))

        return row_labels, col_labels

    def _get_legend_items_per_row(self) -> dict[Any, list[LegendItem]]:
        """Get legend items for each row separately."""
        legend_per_row: dict[Any, list[LegendItem]] = {}

        for row_value in self.row_values:
            row_data_reqs = []
            for col_value in self.col_values:
                cell = self.get_cell_at(row_value, col_value)
                if cell and cell in self.data_reqs_per_cell:
                    row_data_reqs.append(self.data_reqs_per_cell[cell])

            if row_data_reqs:
                legend_per_row[row_value] = self.plot_generator._get_legend_items(relevant_data_reqs_list=row_data_reqs)
            else:
                legend_per_row[row_value] = []

        return legend_per_row

    def _legend_items_to_comparable(self, items: list[LegendItem]) -> frozenset[tuple[str, str, str]]:
        return frozenset((item.label, item.color, item.linestyle) for item in items)

    def _legend_items_differ_by_row(self) -> bool:
        legend_per_row = self._get_legend_items_per_row()
        if len(legend_per_row) <= 1:
            return False
        comparable_legends = [self._legend_items_to_comparable(items) for items in legend_per_row.values()]
        return len(set(comparable_legends)) > 1

    def generate_combined_plot(self, recreate_plots: bool, grid_params: ImageGridParams) -> Image | None:
        """Generate all plots and combine them into a single image."""
        image_grid: list[list[Optional[Path]]] = []
        row_labels, col_labels = self.get_labels()

        for row_value in self.row_values:
            row_images: list[Optional[Path]] = []
            for col_value in self.col_values:
                cell = self.get_cell_at(row_value, col_value)
                if cell:
                    img_path = self.plot_generator.plot_cell_to_path(
                        self.data_reqs_per_cell[cell],
                        cell,
                        recreate_plots,
                    )
                    row_images.append(img_path)
                else:
                    row_images.append(None)
            image_grid.append(row_images)

        filtered_grid = [[path for path in row if path is not None] for row in image_grid]
        filtered_grid = [row for row in filtered_grid if row]

        grid_specific_data_reqs = [
            self.data_reqs_per_cell[cell] for cell in self.cells if cell in self.data_reqs_per_cell
        ]

        if self._legend_items_differ_by_row():
            legend_per_row = self._get_legend_items_per_row()
            legend_items = {
                row_idx: legend_per_row[row_value]
                for row_idx, row_value in enumerate(self.row_values)
                if row_value in legend_per_row
            }
        else:
            legend_items = self.plot_generator._get_legend_items(relevant_data_reqs_list=grid_specific_data_reqs)

        return combine_image_grid(
            filtered_grid,
            grid_params,
            legend_items=legend_items,
            col_labels=col_labels,
            row_labels=row_labels,
        )


def generate_plots_from_plan(
    plot_plan: PlotPlan,
    result_bank: ResultBank,
    recreate_plots: bool = False,
    save_combined_plot: bool = True,
) -> None:
    """
    High-level function to generate plots for a given plot plan.
    """
    plot_generator = PlotGenerator(plot_plan=plot_plan, result_bank=result_bank)
    data_reqs_per_cell = plot_plan.get_data_requirements_per_cell(result_bank)

    cells_by_grid: dict[Any, list[Cell]] = {}
    for cell in data_reqs_per_cell:
        cells_by_grid.setdefault(cell.grids, []).append(cell)

    grid_names = plot_plan.get_options_for_param(FinalPlotsPlanOrientation.grids)
    if not grid_names:
        grid_names = [None]

    for grid_name in grid_names:
        grid_layout = GridLayout(
            plot_plan=plot_plan,
            cells=cells_by_grid[grid_name],
            data_reqs_per_cell=data_reqs_per_cell,
            plot_generator=plot_generator,
        )

        combine_config = plot_plan.combine_plot_config
        if isinstance(combine_config, BaseModel):
            combine_config = combine_config.model_dump()
        # If already a dict, use as-is
        combine_config = ImageGridParams.model_validate(combine_config)

        combined_image = grid_layout.generate_combined_plot(recreate_plots, combine_config)

        if combined_image and save_combined_plot:
            grid_hpd = plot_plan.get_orientation_value_hpd(FinalPlotsPlanOrientation.grids)
            if grid_hpd is not None:
                grid_display_name = grid_hpd.get_display_name(grid_name)
            else:
                grid_display_name = grid_name

            cache_dir = PlotPlans.get_plot_plan_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / f"{plot_plan.plot_id}_{grid_display_name}.png"

            save_at_dpi(combined_image, str(path))
            print(f"Saved combined plot to {path}")
