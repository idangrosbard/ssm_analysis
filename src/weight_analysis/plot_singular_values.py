import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot(df: pd.DataFrame) -> go.Figure:
    return px.line(
        data_frame=df,
        x="rank",
        y="singular_value",
        color="layer",
        title="Singular values per layer",
    )
