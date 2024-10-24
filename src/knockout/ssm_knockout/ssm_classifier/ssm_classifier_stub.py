from .ssm_classifier import SSMClassifier
# from transformers import
from typing import Dict, Iterable
import torch


class SSMClassifierStub(SSMClassifier):
    def __init__(self):
        pass

    def classify_layer(self, layer: torch.nn.Module) -> Dict[str, Iterable[int]]:
        mask = torch.arange(layer.A_log.shape[0])
        return {'all': mask}
