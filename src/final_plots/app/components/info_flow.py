import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.final_plots.app.texts import INFO_FLOW_ANALYSIS_TEXTS
from src.types import TInfoFlowOutput
from src.utils.streamlit_utils import StreamlitComponent


class InfoFlowAnalysisComponent(StreamlitComponent):
    def __init__(self, info_flow_output: TInfoFlowOutput):
        self.info_flow_output = info_flow_output

    def create_interactive_visualization(
        self,
        title: str,
        base_probs: np.ndarray,
        probs_per_window: pd.DataFrame,
        correct_per_window: pd.DataFrame,
    ):
        """Create an interactive visualization with Plotly for better performance."""
        # Get window indices and sort them
        window_indices = sorted([int(col) for col in probs_per_window.columns])

        # Calculate accuracy for each window
        correct_array = np.array(correct_per_window)
        accuracy = correct_array.mean(axis=0)

        # Create a figure with slider
        fig = go.Figure()

        # Add initial scatter plot
        first_window = window_indices[0]
        colors = ["green" if c else "red" for c in correct_per_window[first_window]]

        fig.add_trace(
            go.Scatter(
                x=base_probs,
                y=probs_per_window[first_window],
                mode="markers",
                marker=dict(size=10, color=colors, line=dict(width=1, color="black")),
                name=f"Window {first_window}",
            )
        )

        # Add diagonal reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="black", width=1),
                name="y=x",
                opacity=0.5,
            )
        )

        # Create frames for animation
        frames = []
        for window_idx in window_indices:
            window_str = window_idx
            colors = ["green" if c else "red" for c in correct_per_window[window_str]]

            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=base_probs,
                        y=probs_per_window[window_str],
                        mode="markers",
                        marker=dict(size=10, color=colors, line=dict(width=1, color="black")),
                        name=f"Window {window_str}",
                    ),
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        line=dict(dash="dash", color="black", width=1),
                        name="y=x",
                        opacity=0.5,
                    ),
                ],
                name=str(window_idx),
            )
            frames.append(frame)

        fig.frames = frames

        # Add slider and buttons
        sliders = [
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(font=dict(size=16), prefix="Window: ", visible=True, xanchor="right"),
                transition=dict(duration=300, easing="cubic-in-out"),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[
                            [window_idx],
                            dict(
                                frame=dict(duration=300, redraw=True), mode="immediate", transition=dict(duration=300)
                            ),
                        ],
                        label=f"{window_idx} ({accuracy[i]:.1%} acc)",
                        method="animate",
                    )
                    for i, window_idx in enumerate(window_indices)
                ],
            )
        ]

        # Add play and pause buttons
        updatemenus = [
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=500, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=300, easing="quadratic-in-out"),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0)),
                        ],
                    ),
                ],
            )
        ]

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Base Probability",
            yaxis_title="Knockout Probability",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            updatemenus=updatemenus,
            sliders=sliders,
            height=600,
            width=800,
            showlegend=False,
            hovermode="closest",
        )

        return fig

    def render_probability_distribution(self):
        """Render info flow over time analysis."""
        # Create columns for metrics
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Accuracy Over Time")
            # Extract hit data for each window
            hit_data = {}
            for window_idx, window_data in self.info_flow_output.items():
                hit_data[window_idx] = window_data["hit"]
            accuracy_df = pd.DataFrame(hit_data).mean()

            # Create a Plotly line chart for accuracy
            fig = px.line(
                x=accuracy_df.index,
                y=accuracy_df.values,
                labels={"x": "Window Index", "y": "Accuracy"},
                title=INFO_FLOW_ANALYSIS_TEXTS.accuracy_over_windows,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Add accuracy statistics
            mean_acc = accuracy_df.mean()
            max_acc = accuracy_df.max()
            min_acc = accuracy_df.min()
            st.write(f"Mean Accuracy: {mean_acc:.2%}")
            st.write(f"Max Accuracy: {max_acc:.2%}")
            st.write(f"Min Accuracy: {min_acc:.2%}")

        with col2:
            st.write("### Probability Changes")
            # Extract probability data for each window
            true_probs = {}
            base_probs = {}
            for window_idx, window_data in self.info_flow_output.items():
                true_probs[window_idx] = window_data["true_probs"]
                # Use diffs as a proxy for base_probs if available
                if "diffs" in window_data:
                    diffs = window_data["diffs"]
                    # Approximate base_probs from true_probs and diffs
                    base_probs[window_idx] = [tp - d for tp, d in zip(window_data["true_probs"], diffs)]
                else:
                    # If no diffs, use zeros as placeholder
                    base_probs[window_idx] = [0] * len(window_data["true_probs"])

            true_probs_df = pd.DataFrame(true_probs)
            base_probs_df = pd.DataFrame(base_probs)
            prob_diffs = true_probs_df - base_probs_df

            # Create a Plotly line chart for probability differences
            fig = px.line(
                x=prob_diffs.mean().index,
                y=prob_diffs.mean().values,
                labels={"x": "Window Index", "y": "Probability Difference"},
                title="Probability Changes Over Windows",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Add probability statistics
            mean_diff = prob_diffs.mean().mean()
            max_diff = prob_diffs.max().max()
            min_diff = prob_diffs.min().min()
            st.write(f"Mean Probability Change: {mean_diff:.2%}")
            st.write(f"Max Probability Change: {max_diff:.2%}")
            st.write(f"Min Probability Change: {min_diff:.2%}")

    def render_info_flow_over_time(self):
        """Render probability distribution analysis with interactive visualization."""
        st.write(f"### {INFO_FLOW_ANALYSIS_TEXTS.probability_distribution}")

        # Get sample data for visualization
        window_indices = sorted(list(self.info_flow_output.keys()))

        # Extract data for visualization
        true_probs_data = {}
        hit_data = {}
        for window_idx in window_indices:
            window_data = self.info_flow_output[window_idx]
            true_probs_data[window_idx] = window_data["true_probs"]
            hit_data[window_idx] = window_data["hit"]

        # Create DataFrames
        true_probs_df = pd.DataFrame(true_probs_data)
        hit_df = pd.DataFrame(hit_data)

        # Create base probabilities (this is a simplification)
        first_window_data = self.info_flow_output[window_indices[0]]
        base_probs = np.array([0.5] * len(first_window_data["true_probs"]))

        if "diffs" in first_window_data:
            # If we have diffs, we can calculate base_probs more accurately
            diffs = first_window_data["diffs"]
            true_probs = first_window_data["true_probs"]
            base_probs = np.array([tp - d for tp, d in zip(true_probs, diffs)])

        # Create interactive visualization
        fig = self.create_interactive_visualization(
            "Probability Distribution Over Windows", base_probs, true_probs_df, hit_df
        )

        # Display the interactive visualization
        st.plotly_chart(fig, use_container_width=True)

    def render(self):
        # Display analysis
        st.subheader("Analysis")

        plan = {
            INFO_FLOW_ANALYSIS_TEXTS.TAB_PROBABILITY_DISTRIBUTION: self.render_probability_distribution,
            INFO_FLOW_ANALYSIS_TEXTS.TAB_INFO_FLOW_OVER_TIME: self.render_info_flow_over_time,
        }
        # Tabs for different analysis views
        # tab = st.tabs([tab_name for tab_name in plan])
        tab = [st.expander(tab_name) for tab_name in plan]

        for i, render_func in enumerate(plan.values()):
            with tab[i]:
                with st.spinner("Loading...", show_time=True):
                    render_func()
