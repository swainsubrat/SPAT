"""
A notebook created on 26th December, to edit the execute attack function for all the attacks.
This is as per the new standard.
"""
import os
import argparse
import datetime

import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from art.attacks.evasion import DeepFool, FastGradientMethod
from art.estimators.classification import PyTorchClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from typing import Callable, Tuple, Dict
from pathlib import Path
import pytorch_lightning as pl
from torch import nn

from dataloader import load_mnist
from models.autoencoder import (ANNAutoencoder, BaseAutoEncoder,
                                CelebAAutoencoder, CIFAR10Autoencoder, CIFAR10VAE,
                                CIFAR10LightningAutoencoder)
from models.classifier import (CelebAClassifier, CIFAR10Classifier,
                                MNISTClassifier)

from attacks import ATTACK_MAPPINGS
from attacks.art_attack import get_models, get_xyz, hybridize
from attacks.plot_attack import plot_adversarial_images, plot_robust_accuracy
from dataloader import DATALOADER_MAPPINGS

# class Args:
#     batch_size = 100
#     attack_name = "zoo"
#     device  = "cuda"
#     model_name = "cifar10_cnn_1"
#     ae_name = "cnn_256"
#     plot = False
#     plot_dir = "./plots"
#     # kwargs = {"batch_size": 32, "nb_grads": 5, "epsilon": 1e-05} # deepfool
#     # kwargs = {"eps": 0.1} # pgd and fgsm
#     kwargs = {"batch_size": 30}

# args = Args()

# attack_name = ATTACK_MAPPINGS.get(args.attack_name)
# dataset_name = args.model_name.split("_")[0]
# print(f"Working on the dataset: {dataset_name}!!!!!")

# with open(f"./configs/{dataset_name}.yml", "r") as f:
#     config = yaml.safe_load(f)

# classifier_model, autoencoder_model, config = get_models(args)
# print(f"Loaded classifier and autoencoder models in eval mode!!!!!")
# _, _, test_dataloader = DATALOADER_MAPPINGS[config["dataset_name"]](batch_size=args.batch_size)
# print(f"Loaded dataloader!!!!!")

# x, y, z = get_xyz(args, autoencoder_model, test_dataloader)
    
# config["latent_shape"] = args.ae_name.split('_')[-1]
# classifier, hybrid_classifier, accuracy = hybridize(x, y, z, 
#                                                     config, classifier_model, autoencoder_model)

# # Perform attack
# conditionals = {
#     "calculate_original": True,
#     "is_class_constrained": False
# }

def execute_attack(config, attack_name, x, y, z, classifier, hybrid_classifier, autoencoder_model, kwargs, conditionals):
    result = {}
    name = attack_name.__name__
    result[name] = {}
    
    print(z[1].shape, type(z[1]))
    print(x[1].shape, type(x[1]))
    # ------------------------------------------------- #
    # ---------------- Original Attack ---------------- #
    # ------------------------------------------------- #
    if conditionals["calculate_original"]:
        attack = attack_name(classifier, **kwargs)
        x_adv = attack.generate(x=x[1])
        predictions = classifier.predict(x_test_adv_np)
        x_adv_acc = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])

        result[name]["x_adv"] = x_adv
        result[name]["x_adv_acc"] = x_adv_acc

        # calculate noise
        delta_x = x_test_adv_np - x[1]
        result[name]["delta_x"] = delta_x
        print("Robust accuracy of original adversarial attack: {}%".format(x_adv_acc * 100))
        print("ha ha")

    # ------------------------------------------------- #
    # ---------------- Modified Attack ---------------- #
    # ------------------------------------------------- #
    print(attack_name)
    modified_attack = attack_name(hybrid_classifier, **kwargs)
    if conditionals["is_class_constrained"]:
        z_adv = modified_attack.generate(x=z[1], mask=generate_mask(
            latent_dim=int(config["latent_shape"]),
            n_classes=config["miscs"]["nb_classes"],
            labels=y[1]))
    else:
        z_adv = modified_attack.generate(x=z[1])

    # calculate noise
    x_hat_adv   = autoencoder_model.decoder(torch.Tensor(z_adv).to(config["device"]))
    x_hat       = autoencoder_model.decoder(torch.Tensor(z[1]).to(config["device"]))
    delta_x_hat  = x_hat_adv - x_hat

    # modified attack
    modf_x_adv   = x[1] + delta_x_hat.cpu().detach().numpy()
    predictions = classifier.predict(modf_x_adv)
    modf_x_adv_acc = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])

    result[name]["modf_x_adv"] = modf_x_adv
    result[name]["modf_x_adv_acc"] = modf_x_adv_acc

    # reconstructed attack
    predictions = hybrid_classifier.predict(z_adv)
    x_hat_adv_acc = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])

    result[name]["z_adv"] = z_adv
    result[name]["x_hat_adv"] = x_hat_adv.cpu().detach().numpy()
    result[name]["x_hat_adv_acc"] = x_hat_adv_acc
    
    # send combined noise
    result[name]["delta_x_hat"] = delta_x_hat.cpu().detach().numpy()

    print("Robust accuracy of modified adversarial attack: {}%".format(modf_x_adv_acc * 100))
    print("Robust accuracy of reconstructed adversarial attack: {}%".format(x_hat_adv_acc * 100))

    return result

result: Dict = execute_attack(config, attack_name, x, y, z, classifier, hybrid_classifier, autoencoder_model, args.kwargs, conditionals)
