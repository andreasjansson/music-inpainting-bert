import torch

from config import config

def device():
    if config.gpu:
        return torch.device("cuda")
    return torch.device("cpu")
