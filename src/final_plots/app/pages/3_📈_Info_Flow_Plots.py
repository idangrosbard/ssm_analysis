from itertools import product
from pathlib import Path
from typing import Literal, TypedDict, cast

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.consts import TOKEN_TYPE_COLORS, TOKEN_TYPE_LINE_STYLES
from src.final_plots.data_reqs import (
    get_current_data_reqs,
)
from src.final_plots.results_bank import ParamNames
from src.plots.info_flow_confidence import (
    PlotMetadata,
    create_confidence_plot,
    load_window_outputs,
)

st.set_page_config(page_title="Info Flow Plots", page_icon="📈", layout="wide")

st.title("Info Flow Visualization 📈")


class DataRow(TypedDict):
    experiment_name: str
    model_arch: str
    model_size: str
    window_size: int
    is_all_correct: bool
    source: str
    feature_category: str
    target: str
    data_path: Path


# Cache the data loading
@st.cache_data
def load_info_flow_data() -> list[DataRow]:
    """Load info flow requirements and their fulfillment data"""
    # Get the latest data fulfillments and merge with overrides
    current_fulfilled = get_current_data_reqs()

    data: list[DataRow] = []
    for req, data_path in current_fulfilled.items():
        if not hasattr(req, "experiment_name") or req.experiment_name.value != "info_flow":
            continue

        if not data_path:  # Skip requirements with no fulfillment
            continue

        row_dict = {
            param: getattr(req, param, None)
            for param in ParamNames
            if param not in [ParamNames.path, ParamNames.variation]
        }
        row: DataRow = {**row_dict, "data_path": data_path}  # type: ignore
        data.append(row)

    return data


# Load the data
df = pd.DataFrame(load_info_flow_data())

# Sidebar controls for plot parameters
st.sidebar.header("Plot Configuration")

# Available parameters
available_params = [
    ParamNames.model_arch,
    ParamNames.model_size,
    ParamNames.window_size,
    ParamNames.is_all_correct,
    ParamNames.source,
    ParamNames.target,
    # ParamNames.feature_category,
]

# Parameter roles for plotting
ParamRole = Literal["grid", "column", "row", "line", "fixed"]
PARAM_ROLES: list[ParamRole] = ["fixed", "grid", "column", "row", "line"]

# Initialize session state for parameter roles if not exists
if "param_roles" not in st.session_state:
    st.session_state.param_roles = {param: cast(ParamRole, "fixed") for param in available_params}
    # Set default roles
    st.session_state.param_roles[ParamNames.model_arch] = cast(ParamRole, "grid")
    st.session_state.param_roles[ParamNames.model_size] = cast(ParamRole, "column")
    st.session_state.param_roles[ParamNames.window_size] = cast(ParamRole, "row")
    st.session_state.param_roles[ParamNames.source] = cast(ParamRole, "line")

# Parameter configuration section
st.sidebar.subheader("Parameter Configuration")

# Store selected values for each parameter
param_values = {}


# Function to get available values for a parameter
def get_param_values(param: str) -> list:
    return sorted(df[param].unique())


# Create parameter controls
for param in available_params:
    st.sidebar.markdown(f"**{param}:**")
    col1, col2 = st.sidebar.columns([2, 1])

    with col1:
        # If parameter is fixed, show value selector
        unique_values = get_param_values(param)
        if st.session_state.param_roles[param] == "fixed":
            param_values[param] = st.selectbox(
                f"Value for {param}", unique_values, key=f"value_{param}", label_visibility="collapsed"
            )
        else:
            # Show available values but disabled
            values_str = ", ".join(str(v) for v in unique_values)
            st.text_input(
                f"Available values for {param}", value=values_str, disabled=True, label_visibility="collapsed"
            )

    with col2:
        current_role = st.session_state.param_roles[param]
        selected_role = st.selectbox(
            f"Role for {param}",
            options=PARAM_ROLES,
            # index=PARAM_ROLES.index(current_role),
            key=f"role_{param}",
            label_visibility="collapsed",
        )
        st.session_state.param_roles[param] = cast(ParamRole, selected_role)

# Validate and update roles
role_counts = {role: 0 for role in ["grid", "column", "row", "line"]}
for param, role in st.session_state.param_roles.items():
    if role != "fixed":
        role_counts[role] += 1

# Check if we have exactly one parameter for each role
roles_valid = all(count == 1 for count in role_counts.values())
if not roles_valid:
    st.sidebar.error("Please select exactly one parameter for each role (grid, column, row, line)")
    st.stop()

# Get parameters for each role
grid_param = next(param for param, role in st.session_state.param_roles.items() if role == "grid")
col_param = next(param for param, role in st.session_state.param_roles.items() if role == "column")
row_param = next(param for param, role in st.session_state.param_roles.items() if role == "row")
line_param = next(param for param, role in st.session_state.param_roles.items() if role == "line")

# Filter dataframe based on fixed parameters
for param, value in param_values.items():
    df = df[df[param] == value]

if df.empty:
    st.sidebar.error("No data available for the selected parameter values")
    st.stop()

# Plot customization
st.sidebar.header("Plot Customization")
confidence_level = st.sidebar.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
plot_height = st.sidebar.slider("Plot Height", 300, 1000, 400)
plot_width = st.sidebar.slider("Plot Width", 400, 1200, 600)

# Color customization
st.sidebar.header("Color Customization")
use_custom_colors = st.sidebar.checkbox("Use Custom Colors", False)

if use_custom_colors:
    custom_colors = {}
    custom_styles = {}
    unique_values = df[line_param].unique()

    for value in unique_values:
        if pd.notna(value):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                custom_colors[value] = st.color_picker(f"Color for {value}", TOKEN_TYPE_COLORS.get(value, "#000000"))
            with col2:
                custom_styles[value] = st.selectbox(f"Style for {value}", ["-", "--", ":", "-."], index=0)
else:
    custom_colors = TOKEN_TYPE_COLORS
    custom_styles = TOKEN_TYPE_LINE_STYLES

# Show data paths that will be used
st.header("Data Sources")


# Create tree view of data sources
def display_tree():
    # Get all unique values for each parameter
    grid_values = sorted(df[grid_param].unique())

    # Display tree structure
    for grid_val in grid_values:
        grid_df = df[df[grid_param] == grid_val]

        with st.expander(f"🗂 {grid_param} = {grid_val}", expanded=True):
            row_values = sorted(grid_df[row_param].unique())

            for row_val in row_values:
                row_df = grid_df[grid_df[row_param] == row_val]
                st.markdown(f"**└── {row_param} = {row_val}**")

                col_values = sorted(row_df[col_param].unique())
                for col_val in col_values:
                    col_df = row_df[row_df[col_param] == col_val]
                    st.markdown(f"{'&nbsp;' * 4}**└── {col_param} = {col_val}**")

                    for _, row in col_df.iterrows():
                        line_val = row[line_param]
                        if pd.notna(line_val):
                            st.markdown(f"{'&nbsp;' * 7}└── {line_param} = {line_val}")
                            st.markdown(f"{'&nbsp;' * 10}└── `{row['data_path']}`")


# Display data source tree
col1, col2 = st.columns([1, 3])
with col1:
    show_tree = st.checkbox("Show Data Sources Tree", value=True)
    if show_tree:
        st.info(f"Total experiments: {len(df)}")

with col2:
    if show_tree:
        display_tree()


# Function to create plots
def create_grid_plots():
    grid_values = sorted(df[grid_param].unique())
    plots = {}

    for grid_val in grid_values:
        grid_df = df[df[grid_param] == grid_val]

        # Get unique values for rows and columns
        row_values = sorted(grid_df[row_param].unique())
        col_values = sorted(grid_df[col_param].unique())

        # Create figure with subplots
        fig, axes = plt.subplots(
            len(row_values), len(col_values), figsize=(plot_width / 100, plot_height / 100), squeeze=False
        )

        # Create plots for each combination
        for (i, row_val), (j, col_val) in product(enumerate(row_values), enumerate(col_values)):
            subplot_df = grid_df[(grid_df[row_param] == row_val) & (grid_df[col_param] == col_val)]

            if subplot_df.empty:
                continue

            # Group by line parameter and create plot
            targets_window_outputs = {}
            for _, row in subplot_df.iterrows():
                line_val = row[line_param]
                if pd.isna(line_val):
                    continue

                try:
                    window_outputs = load_window_outputs(row["data_path"])
                    targets_window_outputs[line_val] = window_outputs
                except Exception as e:
                    st.error(f"Error loading data for {line_val}: {e}")
                    continue

            if targets_window_outputs:
                plots_meta_data: dict[Literal["acc", "diff"], PlotMetadata] = {
                    "acc": {
                        "title": "Accuracy",
                        "ylabel": "% accuracy",
                        "ylabel_loc": "center",
                        "axhline_value": 100.0,
                        "ylim": (60.0, 105.0),
                    },
                    "diff": {
                        "title": "Normalized change in prediction probability",
                        "ylabel": "% probability change",
                        "ylabel_loc": "top",
                        "axhline_value": 0.0,
                        "ylim": (-50.0, 50.0),
                    },
                }

                fig = create_confidence_plot(
                    targets_window_outputs=targets_window_outputs,
                    confidence_level=confidence_level,
                    title=f"{grid_param}={grid_val}\n{row_param}={row_val}, {col_param}={col_val}",
                    plots_meta_data=plots_meta_data,
                )

                plots[f"{grid_val}_{row_val}_{col_val}"] = fig

    return plots


# Create plots button
if st.button("Generate Plots"):
    with st.spinner("Generating plots..."):
        plots = create_grid_plots()

        # Display plots
        for plot_key, fig in plots.items():
            st.pyplot(fig)
            plt.close(fig)  # Clean up

        st.success("Plots generated successfully!")

# Help text
st.markdown("""
### How to use:
1. Select parameters for grid layout in the sidebar:
   - Grid Parameter: Creates multiple plot grids
   - Column Parameter: Determines columns in each grid
   - Row Parameter: Determines rows in each grid
   - Line Parameter: Determines different lines within each plot
2. Customize plot appearance:
   - Adjust confidence level
   - Modify plot dimensions
   - Customize colors and line styles
3. Review the data sources tree
4. Click "Generate Plots" to create visualizations
""")
