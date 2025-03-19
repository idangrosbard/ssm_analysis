# %%
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import pandas as pd
from pathlib import Path

SOURCES = ["subject", "relation", "last", "first"]
COLORS = {
    "Mamba-1 2.8B": (0, 153, 76),
    "Mamba-2 2.7B": (76, 0, 153),
    "Falcon-Mamba 7B": (204, 102, 0),
    "GPT-2 1.5B": (0, 153, 153),
}

# %%
from typing import Optional


def get_path_old(root: Path, ws: int, source: str, target: str):
    return root / "info_flow_v7" / "ds=counter_fact" / f"ws={ws}" / f"block_{source}_target_{target}" / "outputs.json"


def get_path_new(arch: str, size: str, ws: int, source: str, target: str, features: Optional[str]):
    info_flow_dir = Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/info_flow/v=v3")
    if features is not None:
        return (
            info_flow_dir
            / f"arch={arch}"
            / f"size={size}"
            / "ds=counter_fact"
            / f"ws={ws}"
            / "outputs"
            / f"target={target}"
            / f"source={source}_feature_category={features}.csv"
        )
    else:
        return (
            info_flow_dir
            / f"arch={arch}"
            / f"size={size}"
            / "ds=counter_fact"
            / f"ws={ws}"
            / "outputs"
            / f"target={target}"
            / f"source={source}.csv"
        )


def get_paths_old(ws: int, source: str, target: str):
    info_flow_dir = Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output.old")

    dirs = {
        "Mamba-1 2.8B": info_flow_dir / "state-spaces" / "mamba-2.8B-hf",
        "Mamba-2 2.7B": info_flow_dir / "state-spaces" / "mamba2-2.7B",
        "Falcon-Mamba 7B": info_flow_dir / "tiiuae" / "falcon-mamba-7b",
        # 'GPT-2': info_flow_dir / 'state-spaces' / 'mamba1-2.8B-hf',
    }

    files = {}

    for key in dirs:
        files[key] = get_path_old(dirs[key], ws, source, target)

    return files


def get_paths_new(ws: int, source: str, target: str, features: Optional[str] = None):
    if features is not None:
        return {}
    info_flow_dir = Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/info_flow/v=v3")

    dirs = {
        "GPT-2 1.5B": info_flow_dir / "arch=gpt2" / "size=1.5B",
        # 'Mamba 1': info_flow_dir / 'arch=mamba1' / 'size=2.8B',
        # 'Mamba 2': info_flow_dir / 'arch=mamba2' / 'size=2.7B',
        # 'Falcon Mamba': info_flow_dir / 'arch=mamba1' / 'size=7B-falcon',
    }

    files = {}

    for key in dirs:
        files[key] = (
            dirs[key] / "ds=counter_fact" / f"ws={ws}" / "outputs" / f"target={target}" / f"source={source}.csv"
        )

    return files

# %%
def load_data(input_path: Path, idx: bool = False):
    with open(input_path, "r") as f:
        data = json.load(f)
    df = {"Depth": [], "Probability diff": [], "Correct": []}

    if idx:
        df["original_idx"] = []

    n_layers = len(data)

    for l in range(n_layers):
        curr_layer = data[str(l)]
        if len(curr_layer) == 0:
            continue

        if "correct" in curr_layer:
            n_samples = len(curr_layer["correct"])
        elif "hit" in curr_layer:
            n_samples = len(curr_layer["hit"])

        for i in range(n_samples):
            df["Depth"] += [l / (n_layers - 1)]
            if "probability_diff" in curr_layer:
                df["Probability diff"] += [curr_layer["probability_diff"][i]]
            elif "diffs" in curr_layer:
                df["Probability diff"] += [curr_layer["diffs"][i]]
            else:
                raise ValueError(f"No probability diff in layer: {list(curr_layer.keys())}")

            if "correct" in curr_layer:
                df["Correct"] += [curr_layer["correct"][i]]
            elif "hit" in curr_layer:
                df["Correct"] += [curr_layer["hit"][i]]
            else:
                raise ValueError(f'No "Correct" in layer: {list(curr_layer.keys())}')

            if idx:
                df["original_idx"] += [curr_layer["original_idx"][i]]

            # df['original_idx'] += [curr_layer['original_idx'][i]]

    return pd.DataFrame(df)

# %%
def plot_trend(
    fig: go.Figure,
    joined: pd.DataFrame,
    model: str,
    rgb: tuple[int, int, int],
    column: str = "Model",
    line_dash: str = "solid",
    col: int = 1,
    row: int = 1,
) -> go.Figure:
    filtered = joined[joined[column] == model]

    upper = filtered["Probability diff_mean"] + filtered["Probability diff_ci95"]
    lower = filtered["Probability diff_mean"] - filtered["Probability diff_ci95"]

    name_conversion = {
        "Mamba-1 2.8B": "mamba-2.8B",
        "Mamba-2 2.7B": "mamba2-2.7B",
        "Falcon-Mamba 7B": "falcon-mamba-7B",
        "GPT-2 1.5B": "gpt2-1.5B",
    }

    if model in name_conversion:
        title = name_conversion[model]
    else:
        title = model

    fig.add_trace(
        go.Scatter(
            x=100 * filtered["Depth"],
            y=filtered["Probability diff_mean"],
            line=dict(color=f"rgb{rgb}", dash=line_dash, width=2),
            mode="lines",
            name=title,
            showlegend=((col == 1) & (row == 1)),
        ),
        col=col,
        row=row,
    )
    fig.add_trace(
        go.Scatter(
            x=(100 * filtered["Depth"]).tolist() + (100 * filtered["Depth"][::-1]).tolist(),  # x, then x reversed
            y=upper.tolist() + lower[::-1].tolist(),  # upper, then lower reversed
            fill="toself",
            fillcolor=f"rgba{rgb + (0.2,)}",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        col=col,
        row=row,
    )
    return fig

# %%
def format_fig(fig: go.Figure, n_cols: int, n_rows: int = 1, font=36) -> go.Figure:
    h = 560

    for c in range(n_cols):
        for r in range(n_rows):
            fig.update_yaxes(
                title_text="% Probability diff.",
                titlefont_size=font,
                row=r + 1,
                col=c + 1,
                showticklabels=True,
                showgrid=True,
                zeroline=True,
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                title_standoff=10,
                gridwidth=1,
                gridcolor="Grey",
                tickfont_size=font,
            )
            fig.update_xaxes(
                title_text="% Depth",
                titlefont_size=font,
                row=r + 1,
                col=c + 1,
                showticklabels=True,
                showgrid=True,
                zeroline=True,
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                title_standoff=10,
                gridwidth=1,
                gridcolor="Grey",
                tickfont_size=font,
            )

    fig = fig.update_layout(
        template="plotly_white",
        title=f"",
        width=h * n_cols,
        height=h * n_rows,
        font=dict(size=font, color="black"),
        showlegend=True,
    )
    fig = fig.update_annotations(font_size=font, font_color="black")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.4, xanchor="center", x=0.5))
    fig.update_annotations(yshift=20)

    fig = fig.add_hline(y=0, line_dash="dash", line_color="black")
    return fig

# %%
from tqdm.auto import tqdm
from typing import Optional
from plotly.subplots import make_subplots


def plot_knockout(ws: int, source: str, target: str, show_yaxis_title: bool = True) -> go.Figure:
    sources = SOURCES

    fig = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        subplot_titles=[f"({['a', 'b', 'c', 'd'][i]}) {src}" for i, src in enumerate(sources)],
    )
    for i, source in enumerate(sources):
        paths = get_paths_new(ws, source, target) | get_paths_old(ws, source, target)

        df = []
        for key in paths:
            df.append(load_data(paths[key]))
            df[-1]["Model"] = key

        df = pd.concat(df)

        means = (
            df.groupby(["Depth", "Model"])
            .mean()
            .reset_index()
            .rename(columns={"Probability diff": "Probability diff_mean"})
        )
        ci95 = (
            df.groupby(["Depth", "Model"])
            .apply(lambda x: 1.96 * x["Probability diff"].std() / np.sqrt(len(x)))
            .reset_index(name="Probability diff_ci95")
        )
        joined = means.merge(ci95, on=["Depth", "Model"], suffixes=("_mean", "_ci95"))

        for model in tqdm(joined["Model"].unique()):
            color = COLORS[model]
            fig = plot_trend(fig, joined, model, color, col=(i) + 1, row=1)

    return format_fig(fig, 4)

# %%
fig = plot_knockout(9, "", "last", True)
fig.write_image(
    "/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 1 - architectures/all.png", scale=2
)

# %%
from tqdm.auto import tqdm


def load_df_size(paths):
    df = []
    for key in paths:
        df.append(load_data(paths[key]))
        df[-1]["size"] = key

    df = pd.concat(df)
    return df


def plot_size_knockout(paths: dict[str, dict[str, dict[str, Path]]]) -> go.Figure:
    sources = SOURCES

    fig = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        subplot_titles=[f"({['a', 'b', 'c', 'd'][i]}) {src}" for i, src in enumerate(sources)],
    )

    for i, source in enumerate(sources):
        sorting = []
        df = []
        for model in paths:
            if model == "mamba":
                sizes = ["mamba-130M", "mamba-1.4B", "mamba-2.8B"]
            elif model == "mamba2":
                sizes = ["mamba2-130M", "mamba2-1.3B", "mamba2-2.7B"]
            elif model == "gpt2":
                sizes = ["gpt2-355M", "gpt2-774M", "gpt2-1.5B"]
            else:
                raise ValueError(f"Unknown model: {model}")

            sorting = sorting + sizes

            df.append(load_df_size(paths[model][source]))
            # df[-1]['Model'] = model

        df = pd.concat(df)

        means = (
            df.groupby(["Depth", "size"])
            .mean()
            .reset_index()
            .rename(columns={"Probability diff": "Probability diff_mean"})
        )
        ci95 = (
            df.groupby(["Depth", "size"])
            .apply(lambda x: 1.96 * x["Probability diff"].std() / np.sqrt(len(x)))
            .reset_index(name="Probability diff_ci95")
        )
        joined = means.merge(ci95, on=["Depth", "size"], suffixes=("_mean", "_ci95"))

        for size in sorting:
            BASE = 80
            ALPHA = 0.25
            if size.startswith("mamba2"):
                color = (int(ALPHA * 76 * 2.5), BASE * ALPHA, int(ALPHA * 153 * 2.5))
            elif size.startswith("mamba"):
                color = (0 * ALPHA, int(153 * ALPHA), int(76 * ALPHA))
            elif size.startswith("gpt2"):
                color = (0 * ALPHA, int(ALPHA * 153), int(ALPHA * 153))
            elif size.startswith("Falcon Mamba"):
                color = (int(204 * ALPHA), int(ALPHA * 102), BASE * ALPHA)
            else:
                raise ValueError(f"Unknown size: {size}")

            brightness = sorting.index(size) % 3  # / len(sizes)
            color = tuple(int(c * (1 + 3 * brightness)) for c in color)
            fig = plot_trend(fig, joined, size, color, "size", col=(i) + 1, row=1)

    return format_fig(fig, 4)

# %%
def get_paths_sizes_mamba(root_dir: Path, search: str, ws: int, source: str, target: str):
    paths = {}
    for path in root_dir.glob(f"{search}*"):
        if path.is_dir():
            paths[path.name.replace("-hf", "").replace("1.3b", "1.3B")] = get_path_old(path, ws, source, target)
    return paths

# %%
def get_path_new(arch: str, size: str, ws: int, source: str, target: str):
    info_flow_dir = Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/info_flow/v=v3")
    return (
        info_flow_dir
        / f"arch={arch}"
        / f"size={size}"
        / "ds=counter_fact"
        / f"ws={ws}"
        / "outputs"
        / f"target={target}"
        / f"source={source}.csv"
    )


def get_paths_sizes_gpt(ws: int, source: str, target: str):
    paths = {}
    sizes = ["355M", "774M", "1.5B"]
    for size in sizes:
        paths[f"gpt2-{size}"] = get_path_new("gpt2", size, ws, source, target)
    return paths


mamba2_paths = {}
mamba_paths = {}
gpt2_paths = {}
for source in SOURCES:
    mamba2_paths[source] = get_paths_sizes_mamba(
        Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output.old/state-spaces"),
        "mamba2-",
        9,
        source,
        "last",
    )
    mamba_paths[source] = get_paths_sizes_mamba(
        Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output.old/state-spaces"), "mamba-", 9, source, "last"
    )
    gpt2_paths[source] = get_paths_sizes_gpt(9, source, "last")
paths = {"mamba": mamba_paths}
plot_size_knockout(paths).write_image(
    "/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 2 - sizes/mamba.png", scale=2
)
# plot_size_knockout(paths).show()

paths = {"mamba2": mamba2_paths}
plot_size_knockout(paths).write_image(
    "/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 2 - sizes/mamba2.png", scale=2
)
# plot_size_knockout(paths).show()

paths = {"gpt2": gpt2_paths}
plot_size_knockout(paths).write_image(
    "/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 2 - sizes/gpt2.png", scale=2
)

# %%
paths = {"mamba2": mamba2_paths}
plot_size_knockout(paths).show()

# %%
def plot_all_sizes():
    outpath = Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/idan_plots")
    outpath.mkdir(exist_ok=True)

    for src in tqdm(SOURCES):
        for target in ["last"]:
            mamba_paths = get_paths_sizes_mamba(
                Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output.old/state-spaces"),
                "mamba-",
                9,
                src,
                target,
            )
            paths = {"mamba": mamba_paths}
            fig = plot_size_knockout(paths)
            fig.write_image(str(outpath / f"mamba_size_{src}_{target}.png"))

            mamba2_paths = get_paths_sizes_mamba(
                Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output.old/state-spaces"),
                "mamba2-",
                9,
                src,
                target,
            )
            paths = {"mamba2": mamba2_paths}
            fig = plot_size_knockout(paths)
            fig.write_image(str(outpath / f"mamba2_size_{src}_{target}.png"))

            gpt2_paths = get_paths_sizes_gpt(9, src, target)
            paths = {"gpt2": gpt2_paths}
            fig = plot_size_knockout(paths)
            fig.write_image(str(outpath / f"gpt2_size_{src}_{target}.png"))


plot_all_sizes()

# %%
def plot_all_sizes():
    outpath = Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/idan_plots")
    outpath.mkdir(exist_ok=True)

    for src in tqdm(SOURCES):
        for target in ["last"]:
            mamba_paths = get_paths_sizes_mamba(
                Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output.old/state-spaces"),
                "mamba-",
                9,
                src,
                target,
            )
            paths = {"mamba": mamba_paths}
            fig = plot_size_knockout(paths, src, target, 0.75)
            fig.write_image(str(outpath / f"mamba_size_{src}_{target}.png"))

            mamba2_paths = get_paths_sizes_mamba(
                Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output.old/state-spaces"),
                "mamba2-",
                9,
                src,
                target,
            )
            paths = {"mamba2": mamba2_paths}
            fig = plot_size_knockout(paths, src, target, 0.75)
            fig.write_image(str(outpath / f"mamba2_size_{src}_{target}.png"))

            gpt2_paths = get_paths_sizes_gpt(9, src, target)
            paths = {"gpt2": gpt2_paths}
            fig = plot_size_knockout(paths, src, target, 0.75)
            fig.write_image(str(outpath / f"gpt2_size_{src}_{target}.png"))


# plot_all_sizes()

# %%
from tqdm.auto import tqdm


def load_df_ws(paths):
    df = []
    for key in paths:
        df.append(load_data(paths[key]))
        df[-1]["ws"] = key

    df = pd.concat(df)
    return df


def plot_ws_knockout(src_paths, model, alpha):
    fig = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        subplot_titles=[f"({['a', 'b', 'c', 'd'][i]}) {src}" for i, src in enumerate(SOURCES)],
    )

    for i, src in enumerate(SOURCES):
        paths = src_paths[src]

        sorting = [1, 3, 5, 9, 12, 15]

        df = []

        df = load_df_ws(paths)

        means = (
            df.groupby(["Depth", "ws"])
            .mean()
            .reset_index()
            .rename(columns={"Probability diff": "Probability diff_mean"})
        )
        ci95 = (
            df.groupby(["Depth", "ws"])
            .apply(lambda x: 1.96 * x["Probability diff"].std() / np.sqrt(len(x)))
            .reset_index(name="Probability diff_ci95")
        )
        joined = means.merge(ci95, on=["Depth", "ws"], suffixes=("_mean", "_ci95"))

        for size in sorting:
            if model.startswith("mamba2"):
                color = (int(alpha * 76), 0, int(alpha * 153))
            elif model.startswith("mamba"):
                color = (0, int(153 * alpha), int(76 * alpha))
            elif model.startswith("gpt2"):
                color = (0, int(alpha * 153), int(alpha * 153))
            elif model.startswith("Falcon Mamba"):
                color = (int(204 * alpha), int(alpha * 102), 0)
            else:
                raise ValueError(f"Unknown model: {model}")

            brightness = sorting.index(size)
            color = tuple(int(c * (1 + brightness)) for c in color)
            fig = plot_trend(fig, joined, size, color, "ws", col=(i) + 1, row=1)

    return format_fig(fig, 4)


def plot_all_window_sizes():
    src_paths = {}
    WS = [1, 3, 5, 9, 12, 15]
    for src in tqdm(SOURCES):
        mamba_ws_path = {}
        for ws in WS:
            mamba_ws_path[ws] = get_paths_sizes_mamba(
                Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output.old/state-spaces"),
                "mamba-",
                ws,
                src,
                "last",
            )

        src_paths[src] = mamba_ws_path

    paths = {model: {} for model in ["mamba-130M", "mamba-1.4B", "mamba-2.8B"]}
    for src in src_paths:
        for model in paths:
            paths[model][src] = {ws: src_paths[src][ws][model] for ws in WS}

    for model in paths:
        fig = plot_ws_knockout(paths[model], model, 0.2)
        # fig.show()
        fig.write_image(
            f"/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 4 - window sizes/{model}.png"
        )

    for src in tqdm(SOURCES):
        mamba_ws_path = {}
        for ws in WS:
            mamba_ws_path[ws] = get_paths_sizes_mamba(
                Path("/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output.old/state-spaces"),
                "mamba2-",
                ws,
                src,
                "last",
            )

        src_paths[src] = mamba_ws_path

    paths = {model: {} for model in ["mamba2-130M", "mamba2-1.3B", "mamba2-2.7B"]}
    for src in src_paths:
        for model in paths:
            paths[model][src] = {ws: src_paths[src][ws][model] for ws in WS}

    for model in paths:
        fig = plot_ws_knockout(paths[model], model, 0.2)
        # fig.show()
        fig.write_image(
            f"/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 4 - window sizes/{model}.png"
        )

    for src in tqdm(SOURCES):
        mamba_ws_path = {}
        for ws in WS:
            mamba_ws_path[ws] = get_paths_sizes_gpt(ws, src, "last")

        src_paths[src] = mamba_ws_path

    paths = {model: {} for model in ["mamba2-130M", "mamba2-1.3B", "mamba2-2.7B"]}
    for src in src_paths:
        print(list(paths.keys()), src_paths[src])
        for model in paths:
            paths[model][src] = {ws: src_paths[src][ws][model] for ws in WS}

    for model in paths:
        fig = plot_ws_knockout(paths[model], model, 0.2)
        # fig.show()
        fig.write_image(
            f"/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 4 - window sizes/{model}.png"
        )

        # mamba_ws_path = {}
        # WS = [1, 3, 5, 9, 12, 15]
        # for ws in WS:
        #     mamba_ws_path[ws] = (get_paths_sizes_mamba(Path('/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output.old/state-spaces'), 'mamba2-', ws, src, 'last'))

        # paths = {model: {ws: mamba_ws_path[ws][model] for ws in WS} for model in ['mamba2-130M', 'mamba2-1.3B', 'mamba2-2.7B']}
        # for model in paths:
        #     fig = plot_ws_knockout(paths[model], model, 0.2)
        #     # fig.show()
        #     fig.write_image(f'/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 4 - window sizes/{model}.png')
        # fig.write_image(f'/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/idan_plots/mamba_ws_{model}_{ws}.png')

    # plot_ws_knockout(mamba_ws_path, 'mamba', 'subject', 'last', 0.75).show()


plot_all_window_sizes()

# %%
from tqdm.auto import tqdm
from typing import Optional


def get_path_new(arch: str, size: str, ws: int, source: str, target: str, features: Optional[str], ds: str, v: int):
    info_flow_dir = Path(f"/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/info_flow/v=v{v}")
    if features is not None:
        return (
            info_flow_dir
            / f"arch={arch}"
            / f"size={size}"
            / f"ds={ds}"
            / f"ws={ws}"
            / "outputs"
            / f"target={target}"
            / f"source={source}_feature_category={features}.csv"
        )
    else:
        return (
            info_flow_dir
            / f"arch={arch}"
            / f"size={size}"
            / f"ds={ds}"
            / f"ws={ws}"
            / "outputs"
            / f"target={target}"
            / f"source={source}.csv"
        )


def plot_knockout_features(
    model_paths: dict[str, dict[Optional[str], Path]], filter: bool = False, show_yaxis_label: bool = True
):
    fig = make_subplots(
        cols=3,
        rows=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.13,
        subplot_titles=["(a) mamba-2.8B", "(b) mamba2-2.7B", "(c) falcon-mamba-7B"],
    )

    for i, model in enumerate(["Mamba-1 2.8B", "Mamba-2 2.7B", "Falcon-Mamba 7B"]):
        paths = model_paths[model]

        df = []
        for key in paths:
            df.append(load_data(paths[key], filter))
            if key is not None:
                df[-1]["Feature category"] = key
            else:
                df[-1]["Feature category"] = "All"

        df = pd.concat(df)

        if filter:
            df = df[
                df["original_idx"].isin(
                    pd.read_csv(
                        "/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/data/preprocessed/counter_fact/filterations/all_correct.csv"
                    )["original_idx"]
                )
            ]
            df.drop(columns=["original_idx"], inplace=True)

        means = (
            df.groupby(["Depth", "Feature category"])
            .mean()
            .reset_index()
            .rename(columns={"Probability diff": "Probability diff_mean"})
        )
        ci95 = (
            df.groupby(["Depth", "Feature category"])
            .apply(lambda x: 1.96 * x["Probability diff"].std() / np.sqrt(len(x)))
            .reset_index(name="Probability diff_ci95")
        )
        joined = means.merge(ci95, on=["Depth", "Feature category"], suffixes=("_mean", "_ci95"))

        for category in tqdm(joined["Feature category"].unique()):
            # (96,96,96)
            color = {"All": (0, 0, 153), "Context independent": (153, 0, 0), "Context dependent": (0, 153, 76)}[
                category
            ]  # {'Mamba-1': (0, 153, 76), 'Mamba-2': (76, 0, 153), 'Falcon-Mamba': (204, 102, 0), 'GPT-2': (0,153,153)}[model]
            dash = "solid"  # {'All': 'solid', 'Fast decay': 'dash', 'Slow decay': 'dot'}[category]
            fig = plot_trend(fig, joined, category, color, "Feature category", dash, col=i + 1, row=1)

    return format_fig(fig, 3, 1, 32)


paths = {
    "Mamba-1 2.8B": {
        "All": get_path_new("mamba1", "2.8B", 9, "subject", "last", None, "counter_fact", 3),
        "Context dependent": get_path_new("mamba1", "2.8B", 9, "subject", "last", "FAST_DECAY", "counter_fact", 3),
        "Context independent": get_path_new("mamba1", "2.8B", 9, "subject", "last", "SLOW_DECAY", "counter_fact", 3),
    },
    "Mamba-2 2.7B": {
        "All": get_path_new("mamba2", "2.7B", 9, "subject", "last", None, "counter_fact", 3),
        "Context dependent": get_path_new("mamba2", "2.7B", 9, "subject", "last", "FAST_DECAY", "counter_fact", 3),
        "Context independent": get_path_new("mamba2", "2.7B", 9, "subject", "last", "SLOW_DECAY", "counter_fact", 3),
    },
    "Falcon-Mamba 7B": {
        "All": get_path_new("mamba1", "7B-falcon", 9, "subject", "last", None, "counter_fact", 3),
        "Context dependent": get_path_new("mamba1", "7B-falcon", 9, "subject", "last", "FAST_DECAY", "counter_fact", 3),
        "Context independent": get_path_new(
            "mamba1", "7B-falcon", 9, "subject", "last", "SLOW_DECAY", "counter_fact", 3
        ),
    },
}

plot_knockout_features(paths, False).write_image(
    "/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 3 - features/feature_knockout.png",
    scale=4,
)

# %%
def get_path_new(arch: str, size: str, ws: int, source: str, target: str, features: Optional[str], ds: str, v: int):
    info_flow_dir = Path(f"/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/output/info_flow/v=v{v}")
    if features is not None:
        return (
            info_flow_dir
            / f"arch={arch}"
            / f"size={size}"
            / f"ds={ds}"
            / f"ws={ws}"
            / "outputs"
            / f"target={target}"
            / f"source={source}_feature_category={features}.csv"
        )
    else:
        return (
            info_flow_dir
            / f"arch={arch}"
            / f"size={size}"
            / f"ds={ds}"
            / f"ws={ws}"
            / "outputs"
            / f"target={target}"
            / f"source={source}.csv"
        )


paths = {
    "All": get_path_new("mamba2", "2.7B", 9, "subject", "last", None, "counter_fact", 3),
    "Fast decay": get_path_new("mamba2", "2.7B", 9, "subject", "last", "FAST_DECAY", "counter_fact", 3),
    "Slow decay": get_path_new("mamba2", "2.7B", 9, "subject", "last", "SLOW_DECAY", "counter_fact", 3),
}

# plot_knockout_features(paths).write_image('/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 3 - features/mamba2_feature_knockout.png')

# %%
paths = {
    "All": get_path_new("mamba1", "7B-falcon", 9, "subject", "last", None, "counter_fact_all_correct", 3),
    "Fast decay": get_path_new(
        "mamba1", "7B-falcon", 9, "subject", "last", "FAST_DECAY", "counter_fact_all_correct", 3
    ),
    "Slow decay": get_path_new(
        "mamba1", "7B-falcon", 9, "subject", "last", "SLOW_DECAY", "counter_fact_all_correct", 3
    ),
}

# plot_knockout_features(paths, 'Falcon Mamba', False, False).write_image('/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 3 - features/falcon_mamba_feature_knockout.png')

# %%


# %%
def plot_shared_knockout(
    model_paths: dict[str, dict[Optional[str], Path]], filter: bool = False, show_yaxis_label: bool = True
):
    fig = make_subplots(
        cols=3,
        rows=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.15,
        subplot_titles=["(a) mamba-2.8B", "(b) mamba2-2.7B", "(c) gpt2-1.5B"],
    )

    for i, model in enumerate(["Mamba-1 2.8B", "Mamba-2 2.7B", "GPT-2 1.5B"]):
        paths = model_paths[model]

        df = []
        for key in paths:
            if paths[key].exists():
                df.append(load_data(paths[key], True))
                df[-1]["Source"] = key
                # df[-1]['Filtered'] = False

        df = pd.concat(df)
        # print(df.shape)

        # df2 = df.copy()
        # df2 = df2[df2['original_idx'].isin(pd.read_csv('/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/data/preprocessed/counter_fact/filterations/all_correct.csv')['original_idx'])]
        # df2['Filtered'] = True
        # print(df2.shape)

        # df = pd.concat([df, df2])

        if filter:
            df = df[
                df["original_idx"].isin(
                    pd.read_csv(
                        "/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/data/preprocessed/counter_fact/filterations/all_correct.csv"
                    )["original_idx"]
                )
            ]
            df.drop(columns=["original_idx"], inplace=True)

        means = (
            df.groupby(["Depth", "Source"])
            .mean()
            .reset_index()
            .rename(columns={"Probability diff": "Probability diff_mean"})
        )
        ci95 = (
            df.groupby(["Depth", "Source"])
            .apply(lambda x: 1.96 * x["Probability diff"].std() / np.sqrt(len(x)))
            .reset_index(name="Probability diff_ci95")
        )
        joined = means.merge(ci95, on=["Depth", "Source"], suffixes=("_mean", "_ci95"))

        # for j, filtered in enumerate([False, True]):
        for source in tqdm(joined["Source"].unique()):
            # color = {'Mamba-1': (0, 153, 76), 'Mamba-2': (76, 0, 153), 'Falcon-Mamba': (204, 102, 0), 'GPT-2': (0,153,153)}[model]
            color = {"first": (0, 153, 76), "relation": (76, 0, 153), "subject": (204, 102, 0), "last": (0, 153, 153)}[
                source
            ]

            fig = plot_trend(fig, joined, source, color, "Source", "solid", col=i + 1, row=1)

    fig = format_fig(fig, n_cols=3, n_rows=1)

    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.4, xanchor="center", x=0.5))
    # fig.update_annotations(yshift=20, xshift=20)

    return fig

# %%
sources = SOURCES

paths = {
    "Mamba-1 2.8B": {src: get_path_new("mamba1", "2.8B", 9, src, "last", None, "counter_fact", 3) for src in sources},
    "Mamba-2 2.7B": {src: get_path_new("mamba2", "2.7B", 9, src, "last", None, "counter_fact", 3) for src in sources},
    "GPT-2 1.5B": {src: get_path_new("gpt2", "1.5B", 9, src, "last", None, "counter_fact", 3) for src in sources},
}


plot_shared_knockout(paths, False).write_image(
    f"/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 6 - all correct/shared_knockout_{False}.png",
    scale=4,
)
plot_shared_knockout(paths, True).write_image(
    f"/home/yandex/DL20232024a/nirendy/repos/ssm_analysis/results/idan_plots/fig 6 - all correct/shared_knockout_{True}.png",
    scale=4,
)

# %%


# %%


# %%


# %%



