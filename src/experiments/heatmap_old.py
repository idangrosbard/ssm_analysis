import numpy as np
import torch
from tqdm import tqdm

from src.datasets.download_dataset import get_hit_dataset
from src.experiments.heatmap import HeatmapConfig
from src.models.model_interface import get_model_interface
from src.utils.logits import get_prompt_row


def main_local(args: HeatmapConfig):
    print(args)
    data = get_hit_dataset(model_id=args.model_id, dataset_args=args.dataset_args)

    args.output_path.mkdir(parents=True, exist_ok=True)

    model_interface = get_model_interface(args.model_arch, args.model_size)
    tokenizer = model_interface.tokenizer
    device = model_interface.device

    n_layers = len(model_interface.model.backbone.layers)

    def forward_eval(prompt_idx, window):
        prompt = get_prompt_row(data, prompt_idx)
        true_id = prompt.true_id(tokenizer, "cpu")
        input_ids = prompt.input_ids(tokenizer, device)

        last_idx = input_ids.shape[1] - 1
        probs = np.zeros((input_ids.shape[1]))

        for idx in range(input_ids.shape[1]):
            num_to_masks = {layer: [(last_idx, idx)] for layer in window}

            next_token_probs = model_interface.generate_logits(
                input_ids=input_ids,
                attention=True,
                num_to_masks=num_to_masks,
            )
            probs[idx] = next_token_probs[0, true_id[:, 0]]
            torch.cuda.empty_cache()
        return probs

    windows = [list(range(i, i + args.window_size)) for i in range(0, n_layers - args.window_size + 1)]

    for prompt_idx in tqdm(args.prompt_indices, desc="Prompts"):
        prob_mat = []
        for window in windows:
            model_interface.setup(layers=window)
            prob_mat.append(forward_eval(prompt_idx, window))

        prob_mat = np.array(prob_mat).T
        np.save(args.output_path / f"idx={prompt_idx}.npy", prob_mat)
