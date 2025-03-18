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

from src.data_defs import DataReqs, PlotPlans, ResultBank
from src.final_plots.app.components.requirements import RequirementExecution, RequirementsDisplay
from src.final_plots.app.texts import FINAL_PLOTS_TEXTS
from src.final_plots.plot_plan import (
    PlotPlan,
    PlotType,
    get_experiment_orientations,
    get_hyper_param_definition,
)
from src.names import EXPERIMENT_NAMES, ExperimentHyperParams, PlotPlanCols, PlotPlanOptionCols, ResultBankParamNames
from src.types import FinalPlotsPlanOrientation, TPlotID
from src.utils.streamlit.aagrid import SelectionMode
from src.utils.streamlit_utils import SessionKey, StreamlitComponent
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
        main_plans = [p for p in self.plot_plans.to_rows() if not p.is_appendix]
        appendix_plans = [p for p in self.plot_plans.to_rows() if p.is_appendix]

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
            if orientation == FinalPlotsPlanOrientation.lines and plan.experiment_name != EXPERIMENT_NAMES.INFO_FLOW:
                continue
            param = plan._get_param_type(orientation)
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
                        if plan.experiment_name == EXPERIMENT_NAMES.INFO_FLOW
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
        self.is_new = plan_id is None
        self.result_bank = result_bank

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
        experiment_name: EXPERIMENT_NAMES,
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
        existing_plan = None
        if self.plan_id:
            existing_plan = self.plot_plans.get_plan(self.plan_id)

        # Form for editing/creating a plot plan
        with st.form("plot_plan_editor"):
            st.subheader("Plot Plan Editor" if self.is_new else "Edit Plot Plan")
            plot_id = st.text_input(
                "Plot ID",
                value="",
                help="Path where the plot will be saved",
            )
            col1, col2 = st.columns([9, 1])
            with col1:
                # Basic information
                title_input = st.text_input(
                    "Title",
                    value="" if self.is_new else existing_plan.title if existing_plan else "",
                    help="Display title for the plot plan",
                )

            with col2:
                order_input = st.number_input(
                    "Order",
                    value=0 if self.is_new else existing_plan.order if existing_plan else 0,
                    help="Order of the plot plan",
                )

            description_input = st.text_area(
                "Description",
                value="" if self.is_new else existing_plan.description if existing_plan else "",
                help="Detailed description of the plot plan",
            )

            # Plot type and experiment
            col1, col2 = st.columns(2)
            with col1:
                plot_type_input = st.selectbox(
                    "Plot Type",
                    options=[pt.name for pt in PlotType],
                    index=0 if self.is_new else list(PlotType).index(existing_plan.plot_type) if existing_plan else 0,
                    help="Type of plot to generate",
                )

            with col2:
                experiment_input = st.selectbox(
                    "Experiment",
                    options=[exp.name for exp in EXPERIMENT_NAMES],
                    index=0
                    if self.is_new
                    else list(EXPERIMENT_NAMES).index(existing_plan.experiment_name)
                    if existing_plan
                    else 0,
                    help="Experiment type for the plot",
                )

            # Appendix flag
            is_appendix_input = st.checkbox(
                "Include in Appendix",
                value=False if self.is_new else existing_plan.is_appendix if existing_plan else False,
                help="Whether this plot should be included in the appendix",
            )

            # Configuration options
            st.subheader("Plot Configuration")

            # Get all available hyperparameters
            hyperparams = [hp.name for hp in ExperimentHyperParams]

            # Get experiment-specific parameters
            experiment_name = EXPERIMENT_NAMES[experiment_input]
            relevant_params = get_experiment_orientations(experiment_name)

            # Parameter selection
            param_values = {}
            for param_type in relevant_params:
                param_name = param_type.value

                # Select parameter type
                param_value = None
                if param_type == FinalPlotsPlanOrientation.rows:
                    rows_input = st.selectbox(
                        "Rows",
                        options=["None"] + hyperparams,
                        index=0
                        if self.is_new or not existing_plan or not existing_plan.rows
                        else hyperparams.index(existing_plan.rows.name) + 1,
                        help="Parameter to vary across rows",
                        key=f"select_{param_name}",
                    )
                    param_value = None if rows_input == "None" else ExperimentHyperParams[rows_input]
                    param_values["rows"] = param_value

                elif param_type == FinalPlotsPlanOrientation.cols:
                    cols_input = st.selectbox(
                        "Columns",
                        options=["None"] + hyperparams,
                        index=0
                        if self.is_new or not existing_plan or not existing_plan.cols
                        else hyperparams.index(existing_plan.cols.name) + 1,
                        help="Parameter to vary across columns",
                        key=f"select_{param_name}",
                    )
                    param_value = None if cols_input == "None" else ExperimentHyperParams[cols_input]
                    param_values["cols"] = param_value

                elif param_type == FinalPlotsPlanOrientation.grids:
                    grids_input = st.selectbox(
                        "Grids",
                        options=["None"] + hyperparams,
                        index=0
                        if self.is_new or not existing_plan or not existing_plan.grids
                        else hyperparams.index(existing_plan.grids.name) + 1,
                        help="Parameter to vary across grid plots",
                        key=f"select_{param_name}",
                    )
                    param_value = None if grids_input == "None" else ExperimentHyperParams[grids_input]
                    param_values["grids"] = param_value

                elif param_type == FinalPlotsPlanOrientation.lines and experiment_name == EXPERIMENT_NAMES.INFO_FLOW:
                    lines_input = st.selectbox(
                        "Lines",
                        options=["None"] + hyperparams,
                        index=0
                        if self.is_new or not existing_plan or not existing_plan.lines
                        else hyperparams.index(existing_plan.lines.name) + 1,
                        help="Parameter to vary across lines in the plot",
                        key=f"select_{param_name}",
                    )
                    param_value = None if lines_input == "None" else ExperimentHyperParams[lines_input]
                    param_values["lines"] = param_value

            st.subheader("Parameter Options")

            options_selected = {
                FinalPlotsPlanOrientation.rows: [],
                FinalPlotsPlanOrientation.cols: [],
                FinalPlotsPlanOrientation.grids: [],
                FinalPlotsPlanOrientation.lines: [],
            }

            # Display option selectors for each parameter
            for param_type in relevant_params:
                param_value = param_values.get(param_type.value)
                if param_value:
                    selected_options, has_options = self._display_option_selector(
                        param_type, existing_plan, experiment_name, param_value
                    )
                    if has_options:
                        options_selected[param_type] = selected_options

            # Submit button
            submit_button = st.form_submit_button("Save Plot Plan")

        # Display option selectors outside the form
        if not submit_button:
            st.markdown("Select specific options for each parameter to limit the plot scope.")

            # Initialize options

            # Display summary
            if any(options_selected.values()):
                st.subheader("Plot Summary")
                rows_count = len(options_selected[FinalPlotsPlanOrientation.rows]) or 1
                cols_count = len(options_selected[FinalPlotsPlanOrientation.cols]) or 1
                grids_count = len(options_selected[FinalPlotsPlanOrientation.grids]) or 1
                lines_count = len(options_selected[FinalPlotsPlanOrientation.lines]) or 1

                total_plots = rows_count * cols_count * grids_count

                st.markdown(f"**Total plots:** {total_plots}")
                st.markdown(f"**Grid structure:** {rows_count} rows × {cols_count} columns × {grids_count} grids")
                if experiment_name == EXPERIMENT_NAMES.INFO_FLOW:
                    st.markdown(f"**Lines per plot:** {lines_count}")

        if submit_button:
            # Validate inputs
            if not title_input:
                st.error("Title is required.")
                return None

            # Convert inputs to appropriate types
            plot_type = PlotType[plot_type_input]
            experiment_name = EXPERIMENT_NAMES[experiment_input]

            # Create the plot plan
            plot_plan = PlotPlan(
                plot_id=TPlotID(plot_id),
                title=title_input,
                description=description_input,
                plot_type=plot_type,
                is_appendix=is_appendix_input,
                order=order_input,
                experiment_name=experiment_name,
                rows=param_values.get("rows"),
                cols=param_values.get("cols"),
                grids=param_values.get("grids"),
                lines=param_values.get("lines") if experiment_name == EXPERIMENT_NAMES.INFO_FLOW else None,
            )

            # Set options for each parameter
            for param_type, options in options_selected.items():
                if options:
                    plot_plan.set_options_for_param(param_type, options)

            return plot_plan

        return None


class PlotPlanRequirements(StreamlitComponent[Optional[DataReqs]]):
    """Component for displaying and managing data requirements for a plot plan."""

    def __init__(self, plot_plan: PlotPlan, result_bank: ResultBank):
        self.plot_plan = plot_plan
        self.result_bank = result_bank

    def render(self) -> Optional[DataReqs]:
        st.subheader("Data Requirements")

        # Get data requirements for the plot plan
        data_reqs = self.plot_plan.get_data_requirements(self.result_bank)
        fulfilled_reqs = data_reqs.to_fulfilled_reqs(self.result_bank).summarize(None)

        if not data_reqs:
            st.info("No data requirements found for this plot plan.")
            return None

        data_reqs_to_run = RequirementsDisplay(
            fulfilled_reqs,
            height=400,
            selection_mode=SelectionMode.MULTIPLE,
            hide_columns=[ResultBankParamNames.is_all_correct],
            key=f"plot_plan_requirements_{self.plot_plan.title}",
        ).render()

        # Option to run missing requirements
        if data_reqs_to_run is not None:
            RequirementExecution(data_reqs_to_run).render()

        return None
