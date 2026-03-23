import torch
from captum.attr import DeepLIFT, IntegratedGradients

def compute_deeplift(model, input_tensor, baseline):
    dl = DeepLIFT(model)
    attributions = dl.attribute(input_tensor, baselines=baseline)
    return attributions
