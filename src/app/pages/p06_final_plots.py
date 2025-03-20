# Purpose: Manage and generate final plots for the report
# High Level Outline:
# 1. Page setup and configuration
# 2. Plot plan management
# 3. Data requirement detection and execution
# 4. Plot generation and export
# Outline Issues:
# - Consider adding batch operations for plot plans
# - Add support for exporting all plots at once
# Outline Compatibility Issues:
# - New file, outline will be implemented

import streamlit as st

from src.data_ingestion.data_defs import PlotPlans
from src.app.components.plot_generation import PlotGenerator
from src.app.components.plot_plans import (
    PlotPlanDetails,
    PlotPlanEditor,
    PlotPlanRequirements,
    PlotPlanSelector,
)
from src.app.components.data_requirements import RequirementExecution
from src.app.data_store import load_results_bank
from src.app.texts import FINAL_PLOTS_TEXTS
from src.analysis.experiment_results.plot_plan import PlotPlan
from src.core.types import TPlotID
from src.utils.streamlit.helpers.component import StreamlitPage
from src.utils.streamlit.helpers.component import StreamlitComponent
from src.utils.streamlit.helpers.session_keys import SessionKeysBase
from src.utils.streamlit.helpers.session_keys import SessionKeyDescriptor

st.set_page_config(page_title=FINAL_PLOTS_TEXTS.title, page_icon=FINAL_PLOTS_TEXTS.icon, layout="wide")

NEW_LABEL = TPlotID("New")


class _FinalPlotsSessionKeys(SessionKeysBase["_FinalPlotsSessionKeys"]):
    SELECTED_PLOT_PLAN_ID = SessionKeyDescriptor[TPlotID](TPlotID(NEW_LABEL))
    EDIT_MODE_KEY = SessionKeyDescriptor[bool](False)
    CONFIRM_RESET = SessionKeyDescriptor[bool](False)

    def is_new_plot_plan(self) -> bool:
        return self.SELECTED_PLOT_PLAN_ID.value == NEW_LABEL


FinalPlotsSessionKeys = _FinalPlotsSessionKeys()


def save_plot_plans(plot_plans: PlotPlans) -> None:
    """Save plot plans to file."""
    plot_plans.save()
    FinalPlotsSessionKeys.EDIT_MODE_KEY.post_external_update(False)
    st.success(FINAL_PLOTS_TEXTS.plot_plans_saved)


class ManagePlotPlans(StreamlitComponent[None]):
    def __init__(self, plot_plans: PlotPlans):
        self.plot_plans = plot_plans

    def render(self):
        # Buttons for adding/editing/deleting plot plans
        col1, col2 = st.columns(2)

        selected_plot_changed = FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.is_changed

        # Plot plan selector
        PlotPlanSelector(
            plot_plans=self.plot_plans,
            selected_plot_id_sk=FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID,
            new_label=NEW_LABEL,
        ).render()

        if selected_plot_changed:
            FinalPlotsSessionKeys.EDIT_MODE_KEY.value = False

        with col1:
            st.checkbox(
                FINAL_PLOTS_TEXTS.edit_plot,
                key=FinalPlotsSessionKeys.EDIT_MODE_KEY.key_for_component,
                disabled=FinalPlotsSessionKeys.is_new_plot_plan(),
            )

        with col2:
            if st.button(
                FINAL_PLOTS_TEXTS.delete_plot,
                use_container_width=True,
                disabled=FinalPlotsSessionKeys.is_new_plot_plan(),
            ):
                self.plot_plans.remove_plan(FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value)
                save_plot_plans(self.plot_plans)
                FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.default_value = NEW_LABEL
                st.rerun()


class FinalPlotsPage(StreamlitPage):
    def render(self):
        # Check if plot plans file exists, if not, create it with default plans
        with st.sidebar:
            load_results_bank.render()

        plot_plans: PlotPlans = PlotPlans.load()
        result_bank = load_results_bank()

        with st.sidebar:
            st.subheader(FINAL_PLOTS_TEXTS.plot_management)
            ManagePlotPlans(plot_plans).render()

        # Main area
        if FinalPlotsSessionKeys.EDIT_MODE_KEY.value or FinalPlotsSessionKeys.is_new_plot_plan():
            # Edit mode
            plot_plan: PlotPlan | None = PlotPlanEditor(
                plot_plans,
                result_bank,
                None if FinalPlotsSessionKeys.is_new_plot_plan() else FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value,
            ).render()

            if st.button(FINAL_PLOTS_TEXTS.cancel):
                FinalPlotsSessionKeys.EDIT_MODE_KEY.post_external_update(False)

            if plot_plan:
                if FinalPlotsSessionKeys.is_new_plot_plan():
                    # Add new plan
                    plot_plans.add_plan(plot_plan)
                else:
                    # Update existing plan
                    plot_plans.remove_plan(FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value)
                    plot_plans.add_plan(plot_plan)

                save_plot_plans(plot_plans)

        elif FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value:
            # Display mode
            selected_plan = plot_plans.get_plan(FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value)
            assert selected_plan is not None
            # Display plan details
            PlotPlanDetails(plot_plans, FinalPlotsSessionKeys.SELECTED_PLOT_PLAN_ID.value, result_bank).render()

            # Display data requirements
            missing_reqs = PlotPlanRequirements(selected_plan, result_bank).render()
            if missing_reqs:
                # Handle requirement execution
                RequirementExecution(missing_reqs).render()

            # Plot generation button
            st.subheader(FINAL_PLOTS_TEXTS.generate_plot)
            plot_path = PlotGenerator(selected_plan, result_bank).render()
            if plot_path:
                st.success(FINAL_PLOTS_TEXTS.plot_saved(plot_path))

        else:
            # No plan selected
            st.info(FINAL_PLOTS_TEXTS.no_plan_selected)


if __name__ == "__main__":
    FinalPlotsPage().render()
