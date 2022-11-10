from art.attacks.evasion import (FastGradientMethod, DeepFool,
                CarliniLInfMethod, BasicIterativeMethod,
                ProjectedGradientDescentPyTorch)

ATTACK_MAPPINGS = {
    "all" : "all",
    "fgsm": FastGradientMethod,
    "pgd" : ProjectedGradientDescentPyTorch,
    # "cnw" : CarliniLInfMethod,
    "bim" : BasicIterativeMethod,
    "deepfool" : DeepFool,
}
