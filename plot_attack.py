"""
File to plot attack results and compare them visually
"""
import yaml
import time
import torch
import numpy as np
import foolbox as fb
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from torch import nn
from torch import Tensor
from typing import Tuple, Callable
from dataloader import load_mnist
from models.classifier import MNISTClassifier, CIFAR10Classifier
from models.autoencoder import ANNAutoencoder, BaseAutoEncoder, CIFAR10Autoencoder
from attack_new import make_hybrid_model, get_embeddings

# first load the config
with open("./configs/mnist.yml", "r") as f:
    config = yaml.safe_load(f)

ROOT             = config["paths"]["root"]
AUTOENCODER_PATH = config["paths"]["autoencoder-path"]
CLASSIFIER_PATH  = config["paths"]["classifier-path"]
BOUNDS           = (config["specs"]["bounds"][0], config["specs"]["bounds"][1])
PLOT             = config["specs"]["plot"]
BATCH_SIZE       = config["specs"]["batch_size"]
RESHAPE          = (config["specs"]["reshape"][0], config["specs"]["reshape"][1])
LOAD_FUNCTION    = load_mnist

epsilons = [i/100 for i in range(1, 50, 1)]
attacks = [
        fb.attacks.FGSM(),
        # fb.attacks.LinfPGD(),
        # fb.attacks.LinfBasicIterativeAttack(),
        # fb.attacks.LinfDeepFoolAttack(),
        # fb.attacks.L2CarliniWagnerAttack(),
        # fb.attacks.LinfAdditiveUniformNoiseAttack(),
    ]

attack_names = [
    "FGSM",
    # "Linf_PGD",
    # "Linf_BIM",
    # "Linf_DeepFool",
    # "L2_C&W",
    # "Linf_AUN"
]

# call appropiate methods to get models and embeddings
fmodel, orig_fmodel, autoencoder_model, classifier_model = make_hybrid_model(
                                            autoencoder_path=AUTOENCODER_PATH,
                                            classifier_path=CLASSIFIER_PATH,
                                            bounds=BOUNDS)
embeddings, images, labels, x_hat = get_embeddings(autoencoder_model=autoencoder_model,
                                                    root=ROOT, 
                                                    load_function=LOAD_FUNCTION,
                                                    batch_size=BATCH_SIZE)

def filter_images(images: Tensor):
    visited = [False] * 10
    ten_integers, ten_lables = [], []
    for img, lbl in zip(images, labels):
        if not visited[lbl.item()]:
            ten_integers.append(img)
            ten_lables.append(lbl)
            visited[lbl.item()] = True
    ten_integers = torch.stack(ten_integers)
    ten_lables   = torch.stack(ten_lables)

    return ten_integers, ten_lables

# launch an attack
# for i, attack in enumerate(attacks):

#     print(f"Running {attack_names[i]} Attack....")

#     # carryout the attack on original
#     start = time.time()
#     orig_noises, orig_advs, orig_success = attack(orig_fmodel, images, labels, epsilons=epsilons)
#     print(f"Time spent on original attack: {time.time() - start}")

#     # carryout the modified attack
#     start = time.time()
#     noises, advs, success = attack(fmodel, embeddings, labels, epsilons=epsilons)
#     print(f"Time spent on modified attack: {time.time() - start}")

# print(type(noises), type(advs), type(success))
# print(noises[-1].shape, advs[-1].shape)
# print(noises[-1][0].shape)