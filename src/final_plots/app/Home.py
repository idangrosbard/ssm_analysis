import streamlit as st

from src.final_plots.app.app_consts import AppSessionKeys
from src.final_plots.app.components.inputs import select_gpu_type, select_window_size
from src.utils.streamlit_utils import StreamlitPage


class HomePage(StreamlitPage):
    def render(self):
        st.set_page_config(page_title="SSM Analysis Results", page_icon="📊", layout="wide")

        st.title("SSM Analysis Results 📊")

        st.markdown("""
        Welcome to the SSM Analysis Results Dashboard! 
        This application provides visualizations and analysis of various experiments conducted on State Space Models.

        ### Available Pages:
        - **📋 Results Bank**: View detailed information about all experiments and their results
        - **📊 Data Requirements**: Manage data requirements and overrides for experiments
        - **📈 Info Flow Plots**: Visualize and analyze information flow with customizable grid layouts
        - **🔥 Heatmap Creation**: Create heatmaps for selected prompts and models
        """)

        st.sidebar.success("Select a page above to explore different analyses.")
        st.session_state

        # Global variables
        st.session_state
        if st.button("Reset App"):
            st.session_state.clear()

        select_gpu_type()
        AppSessionKeys.variation.create_input_widget("Variation")
        select_window_size()


if __name__ == "__main__":
    HomePage().render()
