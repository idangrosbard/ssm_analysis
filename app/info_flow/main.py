import gc
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from app.streamlit_utils import SessionKey
from src.consts import MODEL_SIZES_PER_ARCH_TO_MODEL_ID
from src.datasets.download_dataset import get_hit_dataset
from src.models.model_interface import ModelInterface, get_model_interface
from src.types import (
    DATASETS,
    MODEL_ARCH,
    DatasetArgs,
    TModelID,
    TokenType,
    TPromptData,
)
from src.utils.logits import get_num_to_masks, get_prompt_row

NO_MODEL = "no_model"

# Define available models
MODELS: Dict[Union[str, MODEL_ARCH], Union[str, Dict[str, TModelID]]] = {
    NO_MODEL: {},
    **{
        arch: sizes
        for arch, sizes in MODEL_SIZES_PER_ARCH_TO_MODEL_ID.items()
        if arch
        in [
            MODEL_ARCH.MAMBA1,
            MODEL_ARCH.MINIMAL_MAMBA2_new,
        ]
    },
}


# Define SessionKeys
class SessionKeys:
    selected_model_arch: SessionKey[str] = SessionKey.with_default("selected_model_arch", NO_MODEL)
    selected_model_size: SessionKey[str] = SessionKey("selected_model_size")


def unload_model():
    gc.collect()
    torch.cuda.empty_cache()


# Caching the model loading
@st.cache_resource(max_entries=1, ttl=20)
def load_model_interface(model_arch: MODEL_ARCH, model_size: str) -> ModelInterface:
    print(f"loading model {model_arch} {model_size}")
    unload_model()

    return get_model_interface(model_arch, model_size)


def render_llm_pipeline_hyperparameters():
    # Models and parameters
    st.subheader("Models and Parameters")

    model_arch = st.selectbox(
        "Choose a model architecture",
        options=list(MODELS.keys()),
        format_func=lambda x: x.model_title if isinstance(x, MODEL_ARCH) else x,
        key=SessionKeys.selected_model_arch.key,
    )

    if model_arch != NO_MODEL:
        model_sizes = MODELS[model_arch]  # type: ignore
        if isinstance(model_sizes, dict):
            st.selectbox(
                "Choose model size",
                options=list(model_sizes.keys()),
                key=SessionKeys.selected_model_size.key,
            )

    if st.button("Unload Model"):
        load_model_interface.clear()  # type: ignore
        unload_model()


def run_info_flow_analysis(
    model_interface,
    tokenizer,
    prompt_idx: int,
    window_size: int,
    knockout_src: TokenType,
    knockout_target: TokenType,
    data: TPromptData,
) -> Tuple[List[float], List[float], List[float]]:
    """Run info flow analysis for a single prompt."""
    device = model_interface.device
    n_layers = len(model_interface.model.backbone.layers)
    windows = [list(range(i, i + window_size)) for i in range(0, n_layers - window_size + 1)]

    hits = []
    diffs = []
    true_probs = []

    for window in windows:
        model_interface.setup(layers=window)
        prompt = get_prompt_row(data, prompt_idx)
        num_to_masks, first_token = get_num_to_masks(prompt, tokenizer, window, knockout_src, knockout_target, device)

        next_token_probs = model_interface.generate_logits(
            input_ids=prompt.input_ids(tokenizer, device),
            attention=True,
            num_to_masks=num_to_masks,
        )

        max_prob = np.max(next_token_probs, axis=1)[0]
        true_id = prompt.true_id(tokenizer, "cpu")
        base_prob = prompt.base_prob
        true_prob = next_token_probs[0, true_id[:, 0]]

        hits.append(float(true_prob == max_prob))
        diffs.append(float(((true_prob - base_prob) / base_prob) * 100.0))
        true_probs.append(float(true_prob))

        torch.cuda.empty_cache()

    return hits, diffs, true_probs


def plot_info_flow_results(
    hits: List[float],
    diffs: List[float],
    true_probs: List[float],
    window_size: int,
    knockout_src: TokenType,
    knockout_target: TokenType,
):
    """Create plots for info flow results."""
    n_windows = len(hits)
    x = list(range(n_windows))

    # Create accuracy plot
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=x, y=[h * 100 for h in hits], mode="lines+markers", name="Accuracy"))
    fig_acc.update_layout(
        title=f"Accuracy - knocking out flow from {knockout_src} to {knockout_target}",
        xaxis_title="Window Position",
        yaxis_title="% Accuracy",
        yaxis_range=[0, 100],
    )

    # Create probability change plot
    fig_diff = go.Figure()
    fig_diff.add_trace(go.Scatter(x=x, y=diffs, mode="lines+markers", name="Probability Change"))
    fig_diff.update_layout(
        title=f"Normalized Change in Prediction Probability ({knockout_src} → {knockout_target})",
        xaxis_title="Window Position",
        yaxis_title="% Probability Change",
    )

    # Create true probability plot
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(x=x, y=true_probs, mode="lines+markers", name="True Token Probability"))
    fig_prob.update_layout(
        title=f"True Token Probability ({knockout_src} → {knockout_target})",
        xaxis_title="Window Position",
        yaxis_title="Probability",
    )

    return fig_acc, fig_diff, fig_prob


def render_app():
    # Set the page configuration
    st.set_page_config(
        page_title="Information Flow",
        layout="wide",
    )

    # Sidebar content
    with st.sidebar:
        with st.expander("Models and Parameters", expanded=True):
            render_llm_pipeline_hyperparameters()

            # Add info flow parameters
            st.subheader("Info Flow Parameters")
            window_size = st.slider("Window Size", min_value=1, max_value=15, value=9)
            knockout_src = st.selectbox(
                "Knockout Source",
                options=[
                    TokenType.last,
                    TokenType.first,
                    TokenType.subject,
                    TokenType.relation,
                    TokenType.context,
                ],
                format_func=lambda x: x.value,
            )
            knockout_target = st.selectbox(
                "Knockout Target",
                options=[
                    TokenType.last,
                    TokenType.subject,
                    TokenType.relation,
                ],
                format_func=lambda x: x.value,
            )

    # Main content
    st.title("Information Flow Analysis")

    # Model selection check
    model_arch = st.session_state.get(SessionKeys.selected_model_arch.key)
    model_size = st.session_state.get(SessionKeys.selected_model_size.key)

    if model_arch == NO_MODEL or not model_arch or not model_size:
        st.warning("Please select a model from the sidebar to proceed with the analysis.")
        return

    # Dataset sample selection
    st.header("Dataset Sample")
    data = get_hit_dataset(
        model_id=MODEL_SIZES_PER_ARCH_TO_MODEL_ID[model_arch][model_size],
        dataset_args=DatasetArgs(name=DATASETS.COUNTER_FACT, splits="all"),
    )
    cols = st.columns(2)
    with cols[0]:
        st.write("**Dataset Preview:**")
        i_sample = st.selectbox(
            "Select a sample from the dataset",
            options=range(len(data)),
            format_func=lambda i: f"{i}. {data.iloc[i]['prompt']}",
        )
        selected_dataset_record = data.loc[i_sample].to_dict()
    with cols[1]:
        st.write(pd.Series(selected_dataset_record))

    model_interface = load_model_interface(model_arch, model_size)

    if st.button("Run Info Flow Analysis"):
        with st.spinner("Running analysis..."):
            hits, diffs, true_probs = run_info_flow_analysis(
                model_interface=model_interface,
                tokenizer=model_interface.tokenizer,
                prompt_idx=i_sample,
                window_size=window_size,
                knockout_src=knockout_src,
                knockout_target=knockout_target,
                data=data,
            )

            # Create and display plots
            fig_acc, fig_diff, fig_prob = plot_info_flow_results(
                hits=hits,
                diffs=diffs,
                true_probs=true_probs,
                window_size=window_size,
                knockout_src=knockout_src,
                knockout_target=knockout_target,
            )

            st.subheader("Analysis Results")

            # Display plots in columns
            cols = st.columns(3)
            with cols[0]:
                st.plotly_chart(fig_acc, use_container_width=True)
            with cols[1]:
                st.plotly_chart(fig_diff, use_container_width=True)
            with cols[2]:
                st.plotly_chart(fig_prob, use_container_width=True)

            # Display summary statistics
            st.subheader("Summary Statistics")
            stats_cols = st.columns(3)
            with stats_cols[0]:
                st.metric("Average Accuracy", f"{np.mean(hits) * 100:.1f}%")
            with stats_cols[1]:
                st.metric("Average Prob Change", f"{np.mean(diffs):.1f}%")
            with stats_cols[2]:
                st.metric("Average True Prob", f"{np.mean(true_probs):.3f}")


if __name__ == "__main__":
    render_app()
