import numpy as np
from art.attacks.evasion import (BasicIterativeMethod, CarliniL2Method,
                                 DeepFool, ElasticNet, FastGradientMethod,
                                 ProjectedGradientDescentPyTorch)

# from .art_attack import get_xyz, get_models, hybridize, execute_attack
# from .plot_attack import plot_lips, plot_robust_accuracy

ATTACK_MAPPINGS = {
    "fgsm": FastGradientMethod,
    "pgd" : ProjectedGradientDescentPyTorch,
    "cnw" : CarliniL2Method,
    "bim" : BasicIterativeMethod,
    "deepfool" : DeepFool,
    "elastic": ElasticNet,
}

def generate_mask(latent_dim, n_classes, labels):
    """
    Generate mask given latent_dim, number of classes and
    the label of the adversarial example getting crafted.
    """
    unit_per_class = latent_dim // n_classes
    masks = []
    target = 4
    for label in labels:
        mask = [0] * latent_dim
        mask[label * unit_per_class : (label+1) * unit_per_class] = [1] * unit_per_class
        mask[target * unit_per_class : (target+1) * unit_per_class] = [1] * unit_per_class
        masks.append(mask)

    masks = np.array(masks).astype("bool")

    return masks

# __all__ = [
#     "get_xyz",
#     "get_models",
#     "hybridize",
#     "execute_attack",
#     "plot_lpips",
#     "plot_robust_accuracy",
# ]
