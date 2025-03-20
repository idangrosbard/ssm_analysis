import pandas as pd
import streamlit as st

from src.core.consts import TOKEN_TYPE_COLORS, TOKEN_TYPE_LINE_STYLES
from src.app.app_consts import InfoFlowConsts
from src.utils.streamlit.helpers.component import StreamlitComponent


class plotCustomization(StreamlitComponent):
    def __init__(self, df: pd.DataFrame, line_param: str):
        self.df = df
        self.line_param = line_param

    def render(self):
        st.header("Plot Customization")
        plot_config = {
            "confidence_level": st.slider(
                "Confidence Level",
                0.8,
                0.99,
                InfoFlowConsts.DEFAULT_PLOT_CONFIG["confidence_level"],
                0.01,
            ),
            "plot_height": st.slider(
                "Plot Height",
                300,
                1000,
                InfoFlowConsts.DEFAULT_PLOT_CONFIG["plot_height"],
            ),
            "plot_width": st.slider(
                "Plot Width",
                400,
                1200,
                InfoFlowConsts.DEFAULT_PLOT_CONFIG["plot_width"],
            ),
        }

        # Color customization
        st.header("Color Customization")
        use_custom_colors = st.checkbox("Use Custom Colors", False)

        if use_custom_colors:
            custom_colors = {}
            custom_styles = {}
            unique_values = self.df[self.line_param].unique()

            for value in unique_values:
                if pd.notna(value):
                    col1, col2 = st.columns(2)
                    with col1:
                        custom_colors[value] = st.color_picker(
                            f"Color for {value}", TOKEN_TYPE_COLORS.get(value, "#000000")
                        )
                    with col2:
                        custom_styles[value] = st.selectbox(
                            f"Style for {value}", InfoFlowConsts.DEFAULT_LINE_STYLES, index=0
                        )
        else:
            custom_colors = TOKEN_TYPE_COLORS
            custom_styles = TOKEN_TYPE_LINE_STYLES

        return plot_config, custom_colors, custom_styles
