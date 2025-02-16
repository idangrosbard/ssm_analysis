import streamlit as st
import torch

from src.final_plots.app.app_consts import AppSessionKeys
from src.final_plots.app.components.inputs import select_gpu_type, select_variation, select_window_size

torch.classes.__path__ = []

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

# Global variables


select_gpu_type()

select_variation()

st.write(AppSessionKeys.variation.get())

select_window_size()

st.session_state
