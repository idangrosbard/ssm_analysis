import pandas as pd
from transformers import MambaForCausalLM
import plotly.express as px

from tqdm import tqdm


if __name__ == '__main__':
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
    min_val_per_ssm = df.groupby(['ssm_id', 'layer'])['val'].min().reset_index()
    
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