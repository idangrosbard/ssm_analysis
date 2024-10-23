import pandas as pd
from transformers import MambaForCausalLM
import plotly.express as px

from tqdm import tqdm


def summarize_and_plot(df, model_size):
    # get min val
    min_vals = df.groupby(['ssm_id', 'layer'])['val'].min().reset_index()
    max_vals = df.groupby(['ssm_id', 'layer'])['val'].median().reset_index()
    min_max_vals = min_vals.merge(max_vals, on=['ssm_id', 'layer'], suffixes=('_min', '_max'))
    
    # plot scatter:
    min_max_vals['layer'] = min_max_vals['layer'].astype(str)
    n_layers = len(min_max_vals['layer'].unique())
    colorscale = px.colors.sample_colorscale(px.colors.sequential.Plasma, [i / n_layers for i in range(n_layers)])
    fig = px.scatter(min_max_vals, x='val_min', y='val_max', color='layer', color_discrete_sequence=colorscale)
    fig.update_layout(height=1000, width=1000)
    fig.write_html(f"a_vals_min_max_categ_{model_size}.html")

    
    import torch
    min_max_vals['val_min'] = -torch.exp(torch.from_numpy(min_max_vals['val_min'].to_numpy())).numpy()
    min_max_vals['val_max'] = -torch.exp(torch.from_numpy(min_max_vals['val_max'].to_numpy())).numpy()
    fig = px.scatter(min_max_vals, x='val_min', y='val_max', color='layer', color_discrete_sequence=colorscale)
    fig.update_layout(height=1000, width=1000)
    fig.write_html(f"a_vals_min_max_categ_{model_size}_exp.html")


    min_max_vals['val_min'] = torch.exp(torch.from_numpy(min_max_vals['val_min'].to_numpy())).numpy()
    min_max_vals['val_max'] = torch.exp(torch.from_numpy(min_max_vals['val_max'].to_numpy())).numpy()
    fig = px.scatter(min_max_vals, x='val_min', y='val_max', color='layer', color_discrete_sequence=colorscale)
    fig.update_layout(height=1000, width=1000)
    fig.write_html(f"a_vals_min_max_categ_{model_size}_exp_exp.html")

def summarize_and_plot_skew(df, model_size):
    
    # get skewness
    skewness = df.groupby(['ssm_id', 'layer'])['val'].skew().reset_index()
    # get mode
    mode = df.groupby(['ssm_id', 'layer'])['val'].apply(lambda x: x.mode().values[0]).reset_index()
    # kurtosis
    kurtosis = df.groupby(['ssm_id', 'layer'])['val'].apply(pd.DataFrame.kurt)
    summarized = skewness.merge(mode, on=['ssm_id', 'layer'], suffixes=('_skew', '_mode'))
    summarized = summarized.merge(kurtosis, on=['ssm_id', 'layer'], suffixes=('', '_kurtosis'))

    # plot scatter:
    summarized['layer'] = summarized['layer'].astype(str)
    n_layers = len(summarized['layer'].unique())

    summarized['layer'] = summarized['layer'].astype(str)
    n_layers = len(summarized['layer'].unique())
    colorscale = px.colors.sample_colorscale(px.colors.sequential.Plasma, [i / n_layers for i in range(n_layers)])
    fig = px.scatter_3d(summarized, x='val_skew', y='val_mode', z='val_kurtosis', color='layer', color_discrete_sequence=colorscale)
    fig.update_layout(height=1000, width=1000)
    fig.write_html(f"a_vals_categ_{model_size}_skew_mode_kurt.html")


def summarize_and_plot_norm(df, model_size):
    
    # get powers
    import torch
    df['val'] = torch.exp(-torch.exp(torch.from_numpy(df['val'].to_numpy()))).numpy()
    # get L1 norm
    l1 = df.groupby(['ssm_id', 'layer'])['val'].apply(lambda x: x.abs().sum()).reset_index()
    # get L infinity norm
    linf = df.groupby(['ssm_id', 'layer'])['val'].apply(lambda x: x.abs().max()).reset_index()
    summarized = l1.merge(linf, on=['ssm_id', 'layer'], suffixes=('_l1', '_linf'))

    # print correlation:
    
    
    corrs = summarized.groupby('layer').apply(lambda x: x['val_l1'].corr(x['val_linf'])).reset_index()
    px.line(corrs, x='layer', y=0).write_html(f"a_vals_layer_corrs.html")
    summarized['layer'] = summarized['layer'].astype(str)
    n_layers = len(summarized['layer'].unique())
    colorscale = px.colors.sample_colorscale(px.colors.sequential.Plasma, [i / n_layers for i in range(n_layers)])
    fig = px.scatter(summarized, x='val_l1', y='val_linf', color='layer', color_discrete_sequence=colorscale)
    fig.update_layout(height=1000, width=1000)
    fig.write_html(f"a_vals_categ_{model_size}_norms.html")


def plot_summarzied():
    model_size = '2.8B' # 130M
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.eval()
    vals = {'layer': [], 'val': [], 'ssm_id': [], 'ssm_id+layer': []}

    for i, layer in tqdm(enumerate(model.backbone.layers)):
        for j in range(layer.mixer.A_log.shape[0]):
            for k in range(layer.mixer.A_log.shape[1]):
                vals['layer'].append(i)
                vals['val'].append(layer.mixer.A_log[j,k].item())
                vals['ssm_id'].append(j)
                vals['ssm_id+layer'].append(f'{i}:{j}')

    df = pd.DataFrame(vals)
    summarize_and_plot_norm(df, model_size)

def kmeans():
    model_size = '2.8B' # 130M
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model.eval()

    
    df = {'val': [], 'cluster': [], 'x': [], 'layer': []}
    for i, layer in tqdm(enumerate(model.backbone.layers)):
        A = layer.mixer.A_log

        # sort each row of A independently
        A = A.sort(dim=1).values

        # k means cluster:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=0).fit(A.detach().numpy())
        # plot vals in ascending order
        import plotly.express as px
        import numpy as np
        
        for k in range(kmeans.cluster_centers_.shape[0]):
            curr_mean = kmeans.cluster_centers_[k]
            sorted = curr_mean #np.sort(curr_mean)
            for j in range(sorted.shape[0]):
                df['layer'].append(i)
                
                df['val'].append(sorted[j])
                df['cluster'].append(k)
                df['x'].append(j)
    df = pd.DataFrame(df)
    df['layer1'] = df['layer'] % 8
    df['layer2'] = df['layer'] // 8
    fig = px.line(df, x='x', y='val', color='cluster', facet_row='layer1', facet_col='layer2')
    fig.update_layout(height=8 * 500, width=8 * 500)
    fig.write_html(f"kmeans_{model_size}.html")


if __name__ == '__main__':
    plot_summarzied()
    # kmeans()
