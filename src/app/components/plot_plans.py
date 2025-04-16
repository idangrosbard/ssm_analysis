# Purpose: Provide components for managing plot plans in the final plots page
# High Level Outline:
# 1. Plot plan management components (add, edit, delete)
# 2. Plot plan display components
# 3. Plot plan execution components
# Outline Issues:
# - Consider adding batch operations for plot plans
# - Add visualization preview for plot plans
# Outline Compatibility Issues:
# - New file, outline will be implemented

from typing import Any, List, Optional, Tuple, TypedDict, Union

import streamlit as st
import streamlit_antd_components as sac

from src.analysis.experiment_results.plot_plan import (
    PlotPlan,
    PlotType,
    get_experiment_orientations,
    get_hyper_param_definition,
)
from src.app.components.prompt_filter import SelectFilterationComponent
from src.app.texts import FINAL_PLOTS_TEXTS
from src.core.names import (
    BaseVariantParamName,
    ExperimentHyperParams,
    ExperimentName,
    FinalPlotsPlanOrientation,
    PlotPlanCols,
    PlotPlanOptionCols,
)
from src.core.types import TPlotID
from src.data_ingestion.data_defs.data_defs import PlotPlans, ResultBank
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.streamlit.helpers.session_keys import SessionKey
from src.utils.types_utils import str_enum_values

# Session keys for plot plans


class PlotPlanSelector(StreamlitComponent[None]):
    """Component for selecting a plot plan from the list of available plans."""

    def __init__(
        self,
        plot_plans: PlotPlans,
        selected_plot_id_sk: SessionKey[TPlotID],
        new_label: TPlotID,
    ):
        self.plot_plans = plot_plans
        self.selected_plot_id_sk = selected_plot_id_sk
        self.new_plot_id = new_label

    def render(self):
        if self.plot_plans.is_empty():
            st.info("No plot plans available. Add a new plot plan to get started.")
            return

        if not self.plot_plans.is_plan_exists(self.selected_plot_id_sk.value):
            self.selected_plot_id_sk.value = self.new_plot_id

        # Group plans by appendix/main
        main_plans = [p for p in self.plot_plans.values() if not p.is_appendix]
        appendix_plans = [p for p in self.plot_plans.values() if p.is_appendix]

        # Create menu items
        menu_items: List[Union[str, dict, sac.MenuItem]] = []

        if self.new_plot_id:
            menu_items.append(sac.MenuItem(self.new_plot_id, icon="plus-circle"))

        if main_plans:
            menu_items.append(sac.MenuItem("Main Plots", icon="graph-up", disabled=True))
            for plan in main_plans:
                menu_items.append(
                    sac.MenuItem(
                        plan.plot_id,
                        icon="file-earmark-bar-graph",
                        description=plan.plot_type.name,
                        tag=plan.experiment_name.name,
                    )
                )

        if appendix_plans:
            menu_items.append(sac.MenuItem("Appendix Plots", icon="journal-code", disabled=True))
            for plan in appendix_plans:
                menu_items.append(
                    sac.MenuItem(
                        plan.plot_id,
                        icon="file-earmark-bar-graph",
                        description=plan.plot_type.name,
                        tag=plan.experiment_name.name,
                    )
                )

        sac.menu(
            items=menu_items,
            format_func=lambda x: self.plot_plans.get_plan(x).title if self.plot_plans.is_plan_exists(x) else x,
            key=self.selected_plot_id_sk.key_for_component,
            return_index=False,
        )


class PlotPlanDetails(StreamlitComponent[None]):
    """Component for displaying the details of a selected plot plan."""

    def __init__(self, plot_plans: PlotPlans, selected_plan_id: Optional[TPlotID], result_bank: ResultBank):
        self.plot_plans = plot_plans
        self.selected_plan_id = selected_plan_id
        self.result_bank = result_bank

    def render(self) -> None:
        if not self.selected_plan_id:
            st.info(FINAL_PLOTS_TEXTS.no_plan_selected)
            return

        plan = self.plot_plans.get_plan(self.selected_plan_id)

        col1, col2 = st.columns([3, 5])
        with col1:
            st.dataframe(
                {
                    col: str(getattr(plan, col))
                    for col in str_enum_values(PlotPlanCols)
                    if col not in str_enum_values(FinalPlotsPlanOrientation) + str_enum_values(PlotPlanOptionCols)
                },
                use_container_width=True,
                column_config={
                    PlotPlanCols.description: st.column_config.TextColumn(
                        width="medium",
                    ),
                },
            )

        class SummaryRow(TypedDict):
            orientation: FinalPlotsPlanOrientation
            param: ExperimentHyperParams
            options_count: int
            options: list[str]

        configuration_data: List[SummaryRow] = []
        for orientation in str_enum_values(FinalPlotsPlanOrientation):
            if orientation == FinalPlotsPlanOrientation.lines and plan.experiment_name != ExperimentName.info_flow:
                continue
            param = plan._get_orientation_value(orientation)
            if param:
                options = plan._get_param_options_col(orientation)
                variation_option = get_hyper_param_definition(param)
                if not options:
                    # get all options from result bank
                    options = variation_option.get_result_bank_options(self.result_bank)
                configuration_data.append(
                    {
                        "orientation": orientation,
                        "param": param,
                        "options_count": len(options),
                        "options": [variation_option.get_display_name(option) for option in options],
                    }
                )

        with col2:
            st.dataframe(configuration_data)

        # Display summary
        summary = plan.get_summary()
        grid_total = 1
        grid_structure_text_parts = []
        for orientation in [
            FinalPlotsPlanOrientation.rows,
            FinalPlotsPlanOrientation.cols,
            FinalPlotsPlanOrientation.grids,
        ]:
            size = max(len(summary[orientation]), 1)
            grid_structure_text_parts.append(f"{size} {orientation.value}")
            grid_total *= size

        with col2:
            st.dataframe(
                {
                    FINAL_PLOTS_TEXTS.total_plots_title: (
                        f"{grid_total} = ({FINAL_PLOTS_TEXTS.grid_structure(grid_structure_text_parts)})"
                    ),
                    **(
                        {
                            FINAL_PLOTS_TEXTS.lines_per_plot_title: str(
                                max(len(summary[FinalPlotsPlanOrientation.lines]), 1)
                            ),
                        }
                        if plan.experiment_name == ExperimentName.info_flow
                        else {}
                    ),
                },
                use_container_width=True,
            )


class PlotPlanEditor(StreamlitComponent[Optional[PlotPlan]]):
    """Component for editing or creating a plot plan."""

    def __init__(self, plot_plans: PlotPlans, result_bank: ResultBank, plan_id: Optional[TPlotID] = None):
        self.plot_plans = plot_plans
        self.plan_id = plan_id
        self.result_bank = result_bank

    @property
    def is_new(self) -> bool:
        return self.plan_id is None

    def _get_options_for_param(self, param_type: Optional[ExperimentHyperParams]) -> List[Any]:
        """Get available options for a parameter type."""
        if not param_type:
            return []

        return list(get_hyper_param_definition(param_type).get_options(self.result_bank))

    def _get_display_names_for_options(
        self, options: List[Any], param_type: Optional[ExperimentHyperParams]
    ) -> List[str]:
        """Get display names for options."""
        if not param_type or not options:
            return []

        variation_option = get_hyper_param_definition(param_type)
        return [variation_option.get_display_name(option) for option in options]

    def _display_option_selector(
        self,
        param_type: FinalPlotsPlanOrientation,
        existing_plan: Optional[PlotPlan],
        experiment_name: ExperimentName,
        param_value: Optional[ExperimentHyperParams],
    ) -> Tuple[List[Any], bool]:
        """Display a multi-select for parameter options."""
        if not param_value:
            return [], False

        # Check if this parameter is relevant for the experiment type
        relevant_params = get_experiment_orientations(experiment_name)
        if param_type not in relevant_params:
            return [], False

        # Get available options
        available_options = self._get_options_for_param(param_value)
        if not available_options:
            return [], False

        # Get display names for options
        display_names = self._get_display_names_for_options(available_options, param_value)
        options_map = {name: option for name, option in zip(display_names, available_options)}

        # Get currently selected options
        selected_options = []
        if existing_plan:
            selected_options = existing_plan.get_options_for_param(param_type)
            selected_display_names = self._get_display_names_for_options(selected_options, param_value)
        else:
            selected_display_names = []

        # Display multi-select
        st.markdown(f"**Select {param_type.value.capitalize()} Options:**")
        selected_names = st.multiselect(
            f"Available {param_value.name} options",
            options=display_names,
            default=selected_display_names,
            help=f"Select specific {param_value.name} values to include in the plot",
            key=f"multiselect_{param_type.value}_{param_value.name}",
        )

        # Convert selected names back to actual options
        selected = [options_map[name] for name in selected_names]

        return selected, True

    def render(self) -> Optional[PlotPlan]:
        # Get the existing plan if editing
        if self.plan_id:
            existing_plan = self.plot_plans.get_plan(self.plan_id)
        else:
            existing_plan = PlotPlan(
                plot_id=TPlotID(""),
                title="",
                description="",
                experiment_name=ExperimentName.info_flow,
                plot_type=PlotType.ARCHITECTURE_KNOCKOUT,
                is_appendix=False,
                order=0,
                fixed_values={},
                cell_plot_config={},
                combine_plot_config={},
            )

        # Form for editing/creating a plot plan
        st.subheader("New Plot Plan" if self.is_new else "Edit Plot Plan")
        for i, col in enumerate(st.columns([3, 3, 1, 1])):
            with col:
                if i == 0:
                    existing_plan.plot_id = TPlotID(
                        st.text_input(
                            "Plot ID",
                            value=existing_plan.plot_id,
                            help="Path where the plot will be saved",
                            disabled=not self.is_new,
                        )
                    )
                elif i == 1:
                    # Basic information
                    existing_plan.title = st.text_input(
                        "Title",
                        value=existing_plan.title,
                        help="Display title for the plot plan",
                    )
                elif i == 2:
                    existing_plan.order = st.number_input(
                        "Order",
                        value=existing_plan.order,
                        help="Order of the plot plan",
                    )
                elif i == 3:
                    existing_plan.is_appendix = st.checkbox(
                        "Appendix",
                        value=existing_plan.is_appendix,
                        help="Whether this plot should be included in the appendix",
                    )

        existing_plan.description = st.text_area(
            "Description",
            value=existing_plan.description,
            help="Detailed description of the plot plan",
        )

        # Plot type and experiment
        for i, col in enumerate(st.columns(2)):
            with col:
                if i == 0:
                    existing_plan.experiment_name = ExperimentName(
                        st.selectbox(
                            "Experiment",
                            options=[exp.name for exp in ExperimentName],
                            index=list(ExperimentName).index(existing_plan.experiment_name),
                            help="Experiment type for the plot",
                        )
                    )
                elif i == 1:
                    existing_plan.plot_type = PlotType(
                        st.selectbox(
                            "Plot Type",
                            options=[pt.name for pt in PlotType],
                            index=list(PlotType).index(existing_plan.plot_type),
                            help="Type of plot to generate",
                        )
                    )

        # Get all available hyperparameters
        hyperparams = [hp.name for hp in ExperimentHyperParams]

        # Get experiment-specific parameters
        orientations = get_experiment_orientations(existing_plan.experiment_name)

        NONE_STR = "None"
        # Parameter selection
        for i, col in enumerate(st.columns(len(orientations))):
            with col:
                orientation = orientations[i]
                param_name = orientation.value
                current_index = 0
                if current_value := existing_plan._get_orientation_value(orientation):
                    current_index = hyperparams.index(current_value.name) + 1
                _orientation_input = st.selectbox(
                    orientation.value.capitalize(),
                    options=[NONE_STR] + hyperparams,
                    index=current_index,
                    help="Parameter to vary across rows",
                    key=f"select_{param_name}",
                )
                orientation_input = (
                    None if _orientation_input == NONE_STR else ExperimentHyperParams[_orientation_input]
                )
                existing_plan.set_orientation_value(orientation, orientation_input)
                if orientation_input:
                    selected_options, has_options = self._display_option_selector(
                        orientation, existing_plan, existing_plan.experiment_name, orientation_input
                    )
                    if has_options:
                        existing_plan.set_options_for_orientation(orientation, selected_options)

        derived_variant_params = existing_plan.get_derived_variants_params()
        missing_cols = [
            col
            for col in ExperimentName.get_variant_cols(existing_plan.experiment_name)
            if (col not in [BaseVariantParamName.experiment_name] and col not in derived_variant_params)
        ]

        for col_name, st_col in zip(missing_cols, st.columns(len(missing_cols))):
            with st_col:
                hpd_col = ExperimentHyperParams(col_name)
                options = get_hyper_param_definition(hpd_col).get_options(self.result_bank)
                existing_plan.fixed_values[hpd_col] = st.selectbox(
                    col_name.capitalize(),
                    options=options,
                    index=options.index(get_hyper_param_definition(hpd_col).default_fix_value()),
                    key=f"select_{col_name}",
                )

        if ExperimentHyperParams.filteration not in derived_variant_params:
            existing_plan.fixed_values[ExperimentHyperParams.filteration] = SelectFilterationComponent(
                key=f"select_{ExperimentHyperParams.filteration}",
                context_model_arch_and_sizes=existing_plan.derive_model_arch_and_sizes_context(),
            ).render()

        # Submit button
        submit_button = st.button("Save Plot Plan")

        # Display option selectors outside the form
        if not submit_button:
            # Display summary
            if any(existing_plan.get_options_for_param(orientation) for orientation in orientations):
                st.subheader("Plot Summary")
                rows_count = len(existing_plan.rows_options) or 1
                cols_count = len(existing_plan.cols_options) or 1
                grids_count = len(existing_plan.grids_options) or 1
                lines_count = len(existing_plan.lines_options) or 1

                total_plots = rows_count * cols_count * grids_count

                st.markdown(f"**Total plots:** {total_plots}")
                st.markdown(f"**Grid structure:** {rows_count} rows × {cols_count} columns × {grids_count} grids")
                if existing_plan.experiment_name == ExperimentName.info_flow:
                    st.markdown(f"**Lines per plot:** {lines_count}")

        if submit_button:
            return existing_plan

        return None
