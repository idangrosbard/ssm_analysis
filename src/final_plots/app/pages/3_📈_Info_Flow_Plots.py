# Purpose: Create and display information flow plots with customizable parameters and visualization options
# High Level Outline:
# 1. Page setup and configuration
# 2. Parameter configuration and role assignment
# 3. Data source management and display
# 4. Plot creation and customization
# Outline Issues:
# - Add export functionality for generated plots
# - Consider adding more plot customization options
# Outline Compatibility Issues:
# - Current implementation follows the outline structure correctly

from typing import cast

import pandas as pd
import streamlit as st

from src.consts import EXPERIMENT_NAMES
from src.final_plots.app.app_consts import InfoFlowConsts
from src.final_plots.app.components.multi_plots import DataSourceDisplay, ParameterConfiguration, PlotCreation
from src.final_plots.app.components.plot_customization import plotCustomization
from src.final_plots.app.data_store import load_experiment_fulfilled_reqs_df
from src.final_plots.app.texts import INFO_FLOW_TEXTS
from src.final_plots.results_bank import ParamNames
from src.utils.streamlit_utils import StreamlitPage

st.set_page_config(page_title=INFO_FLOW_TEXTS.title, page_icon=INFO_FLOW_TEXTS.icon, layout="wide")
st.title(f"{INFO_FLOW_TEXTS.title} {INFO_FLOW_TEXTS.icon}")


class InfoFlowPlotsPage(StreamlitPage):
    def render(self):
        # Load the data
        df = pd.DataFrame(load_experiment_fulfilled_reqs_df(EXPERIMENT_NAMES.INFO_FLOW))

        # Available parameters
        available_params = [
            ParamNames.model_arch,
            ParamNames.model_size,
            ParamNames.window_size,
            ParamNames.is_all_correct,
            ParamNames.source,
            ParamNames.target,
        ]

        # Initialize session state for parameter roles if not exists
        if "param_roles" not in st.session_state:
            st.session_state.param_roles = {
                param: cast(InfoFlowConsts.ParamRole, "fixed") for param in available_params
            }
            # Set default roles
            st.session_state.param_roles[ParamNames.model_arch] = cast(InfoFlowConsts.ParamRole, "grid")
            st.session_state.param_roles[ParamNames.model_size] = cast(InfoFlowConsts.ParamRole, "column")
            st.session_state.param_roles[ParamNames.window_size] = cast(InfoFlowConsts.ParamRole, "row")
            st.session_state.param_roles[ParamNames.source] = cast(InfoFlowConsts.ParamRole, "line")

        # Configure parameters and get roles
        param_values, param_roles = ParameterConfiguration(df, available_params).render()
        if param_values is None or param_roles is None:
            st.stop()

        # Filter dataframe based on fixed parameters
        for param, value in param_values.items():
            df = df[df[param] == value]

        if df.empty:
            st.sidebar.error("No data available for the selected parameter values")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            # Configure plot customization
            plot_config, custom_colors, custom_styles = plotCustomization(df, param_roles["line"]).render()

        with col2:
            # Display data sources
            DataSourceDisplay(df, param_roles).render()

        # Create and display plots
        PlotCreation(df, param_roles, plot_config, custom_colors, custom_styles).render()


if __name__ == "__main__":
    InfoFlowPlotsPage().render()
