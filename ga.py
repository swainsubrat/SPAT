import torch
from typing import Tuple

def generate_perturbation(size: Tuple):
    """
    perturbation is a chromosome same size as the latent space
    dimension.
    """
    chromosome = torch.rand(size)
    return chromosome