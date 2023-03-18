import numpy as np

from art.attacks.evasion import (FastGradientMethod, DeepFool,
                CarliniL2Method, BasicIterativeMethod,
                ProjectedGradientDescentPyTorch, ZooAttack,
                HopSkipJump, SaliencyMapMethod, ElasticNet,
                FeatureAdversariesPyTorch, FrameSaliencyAttack,
                GeoDA, PixelAttack, ThresholdAttack,
                SignOPTAttack, SimBA, SpatialTransformation,
                SquareAttack)

ATTACK_MAPPINGS = {
    "all" : "all",
    "fgsm": FastGradientMethod,
    "pgd" : ProjectedGradientDescentPyTorch,
    "cnw" : CarliniL2Method,
    "bim" : BasicIterativeMethod,
    "deepfool" : DeepFool,
    "zoo": ZooAttack,
    "hopskipjump": HopSkipJump,
    "jsma": SaliencyMapMethod,
    "elastic": ElasticNet,
    # "featureadv": FeatureAdversariesPyTorch,
    # "framesilency": FrameSaliencyAttack,
    # "geoda": GeoDA,
    # "pixel": PixelAttack,
    # "threshold": ThresholdAttack,
    "signopt": SignOPTAttack,
    # "simba": SimBA,
    # "spatial": SpatialTransformation,
    # "square": SquareAttack
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
