import argparse
import datetime
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from art.attacks.evasion import DeepFool, FastGradientMethod
from art.estimators.classification import PyTorchClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import logging
from pathlib import Path
from typing import Callable, Dict, Tuple

import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm

logging.basicConfig(filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info('This will get logged to a file')

from attacks import ATTACK_MAPPINGS, FastGradientMethod
from attacks.art_attack import execute_attack, get_models, get_xyz, hybridize
from attacks.plot_attack import plot_adversarial_images, plot_robust_accuracy
from dataloader import DATALOADER_MAPPINGS, load_mnist
from models.autoencoder import (CIFAR10VAE, ANNAutoencoder, BaseAutoEncoder,
                                CelebAAutoencoder, CIFAR10Autoencoder,
                                CIFAR10LightningAutoencoder,
                                CIFAR10NoisyLightningAutoencoder)
from models.classifier import (CelebAClassifier, CIFAR10Classifier,
                               MNISTClassifier)


class Args:
    batch_size = 64
    dataset_len = 1000
    attack_name = "cnw"
    device  = "cuda"
    model_name = "imagenet_inceptionv3"
    ae_name = "vgg_16"
    plot = False
    plot_dir = "./plots"
    # kwargs = {}
    # kwargs = {"eps": 0.1, "batch_size": 64} # fgsm
    # kwargs = {"batch_size": 128, "nb_grads": 5, "epsilon": 1e-04} # deepfool
    # kwargs = {"eps": 0.003, "batch_size": 64} # pgd and bim
    # kwargs = {"batch_size": 32, "theta": 0.3} # jsma
    kwargs = {"batch_size": 64} # cnw
    # kwargs = {"batch_size": 128, "targeted": False} # boundary and elastic and signopt

args = Args()

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

attack_name = ATTACK_MAPPINGS.get(args.attack_name)
dataset_name = args.model_name.split("_")[0]
logger.info(f"Working on the dataset: {dataset_name}!!!!!")
logger.info("----------------------------- CnW ----------------------------------")

with open(f"./configs/{dataset_name}.yml", "r") as f:
    config = yaml.safe_load(f)

classifier_model, autoencoder_model, config = get_models(args)
logger.info(f"Loaded classifier and autoencoder models in eval mode!!!!!")

train_dataloader = DATALOADER_MAPPINGS[config["dataset_name"]](batch_size=args.batch_size)
logger.info(f"Loaded dataloader!!!!!")

result = {attack_name.__name__: {}}
xs, ys = [], []
cas, ras = [], []
x_adv, x_adv_acc, delta_x = [], [], []
modf_x_adv, modf_x_adv_acc = [], []
z_adv, x_hat_adv, x_hat_adv_acc, delta_x_hat = [], [], [], []
orig_time, modf_time = [], []

for images, labels in tqdm(train_dataloader):
    x_test, y_test = images.to(args.device), labels.to(args.device)
    x_test_np, y_test_np = x_test.cpu().numpy(), y_test.cpu().numpy()

    with torch.no_grad():
        z_test = autoencoder_model.get_z(x_test)
    z_test_np = z_test.detach().cpu().numpy()

    x, y, z = (x_test, x_test_np), (y_test, y_test_np), (z_test, z_test_np)

#     config["latent_shape"] = (512, 7, 7)
#     classifier, hybrid_classifier, ca, ra = hybridize(x, y, z, 
#                                                         config, classifier_model, autoencoder_model)
    xs.append(x[0])
    for ele in y[1]:
        ys.append(ele)
#     cas.append(ca)
#     ras.append(ra)
#     # Perform attack
#     conditionals = {
#         "calculate_original": True,
#         "is_class_constrained": False
#     }
#     results: Dict = execute_attack(config, attack_name, x, y, z, classifier, hybrid_classifier, autoencoder_model, args.kwargs, conditionals)[attack_name.__name__]
#     # results = result[attack_name.__name__]
#     x_adv.append(results["x_adv"])
#     x_adv_acc.append(results["x_adv_acc"])
#     delta_x.append(results["delta_x"])
#     modf_x_adv.append(results["modf_x_adv"])
#     modf_x_adv_acc.append(results["modf_x_adv_acc"])
#     z_adv.append(results["z_adv"])
#     x_hat_adv.append(results["x_hat_adv"])
#     x_hat_adv_acc.append(results["x_hat_adv_acc"])
#     delta_x_hat.append(results["delta_x_hat"])

#     orig_time.append(results["orig_time"])
#     modf_time.append(results["modf_time"])

# logger.info("Accuracy on benign test examples: {}%".format((sum(cas)/len(cas)) * 100))
# logger.info("Accuracy on benign test examples(from reconstructed): {}%".format((sum(ras)/len(ras)) * 100))

# result[attack_name.__name__]["x_adv"] = np.vstack(x_adv)
# result[attack_name.__name__]["x_adv_acc"] = sum(x_adv_acc) / len(x_adv_acc)
# result[attack_name.__name__]["delta_x"] = np.vstack(delta_x)

# result[attack_name.__name__]["modf_x_adv"] = np.vstack(modf_x_adv)
# result[attack_name.__name__]["modf_x_adv_acc"] = sum(modf_x_adv_acc) / len(modf_x_adv_acc)
# result[attack_name.__name__]["z_adv"] = np.vstack(z_adv)
# result[attack_name.__name__]["x_hat_adv"] = np.vstack(x_hat_adv)
# result[attack_name.__name__]["x_hat_adv_acc"] = sum(x_hat_adv_acc) / len(x_hat_adv_acc)
# result[attack_name.__name__]["delta_x_hat"] = np.vstack(delta_x_hat)
xs = torch.vstack(xs)
ys = np.array(ys)

# logger.info("Robust accuracy of original adversarial attack: {}%".format(result[attack_name.__name__]["x_adv_acc"] * 100))
# logger.info("Robust accuracy of modified adversarial attack: {}%".format(result[attack_name.__name__]["modf_x_adv_acc"] * 100))
# logger.info("Robust accuracy of reconstructed adversarial attack: {}%".format(result[attack_name.__name__]["x_hat_adv_acc"] * 100))

# logger.info(f"Time taken for original attack: {sum(orig_time)} seconds")
# logger.info(f"Time taken for modified attack: {sum(modf_time)} seconds")

# import torchvision
# def plot_images(images):
#     plt.figure(figsize=(20, 2))
#     images = torch.Tensor(images).reshape(-1, 3, 32, 32)
#     grid = torchvision.utils.make_grid(images, nrow=10, normalize=True, range=(-1,1))
#     grid = grid.permute(1, 2, 0)
#     plt.imshow(grid)
#     plt.axis('off')
#     plt.show()

# def plot_batch(images):
#     plt.figure(figsize=(20, 12))
#     images = torch.Tensor(images).reshape(-1, 3, 224, 224)
#     grid = torchvision.utils.make_grid(images, nrow=10, normalize=False, range=(0,1))
#     grid = grid.permute(1, 2, 0)
#     plt.imshow(grid)
#     plt.axis('off')
#     plt.savefig(f"./img/{attack_name.__name__}.png", dpi=600)
#     plt.show()

# start = 0
# end   = 10

# images = np.vstack([x[1][start: end], x_adv[start: end], delta_x[start: end], modf_x_adv[start: end], x_hat_adv[start: end], delta_x_hat[start: end]])
# plot_batch(images)

# # save adversarial images
# fileObj = open(f"/home/sweta/scratch/objects/{dataset_name}_{args.attack_name}.pkl", 'wb')
# pickle.dump(result, fileObj)
# fileObj.close()
# logger.info("Saved the adversarial images!!!!")

# load adversarial images
file = open(f"/home/sweta/scratch/objects/{dataset_name}_{args.attack_name}.pkl", 'rb')
result = pickle.load(file)
file.close()

conditionals = {
    "calculate_original": True
}

if conditionals["calculate_original"]:
    x_adv = result[attack_name.__name__]["x_adv"]
    delta_x = result[attack_name.__name__]["delta_x"]

x_hat_adv  = result[attack_name.__name__]["x_hat_adv"]
modf_x_adv = result[attack_name.__name__]["modf_x_adv"]

# noises
delta_x_hat = result[attack_name.__name__]["delta_x_hat"]

import lpips

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

# LPIPS between original and original attacks
import torch

img_orig = torch.Tensor(x_adv) # image should be RGB, IMPORTANT: normalized to [-1,1]
img_modf = torch.Tensor(x_hat_adv)
# img_modf = torch.Tensor(modf_x_adv)
img = xs.detach().cpu()

orig_lpips = loss_fn_alex(img, img_orig)
modf_lpips = loss_fn_alex(img, img_modf)
print("Average LPIPS score of original adversarial attack: ", orig_lpips.flatten().mean())
print("Average LPIPS score of modifed adversarial attack: ", modf_lpips.flatten().mean())

## Harmonic Means
orig_acc = result[attack_name.__name__]["x_adv_acc"]
modf_acc = result[attack_name.__name__]["x_hat_adv_acc"]

orig_lpips_avg = orig_lpips.flatten().mean()
modf_lpips_avg = modf_lpips.flatten().mean()

orig_hm = (orig_acc * orig_lpips_avg) / (orig_acc + orig_lpips_avg)
modf_hm = (modf_acc * modf_lpips_avg) / (modf_acc + modf_lpips_avg)

print(f"Original HM: {orig_hm}, Modified HM: {modf_hm}")

orig_linf = torch.max(torch.abs(xs - img_orig.to(device)))
modf_linf = torch.max(torch.abs(xs - img_modf.to(device)))

print("Average Linf distance between original and original adversarial images: ", orig_linf.mean())
print("Average Linf distance between original and modified adversarial images: ", modf_linf.mean())

orig_l2 = torch.cdist(xs, img_orig.to(device), p=2)
modf_l2 = torch.cdist(xs, img_modf.to(device), p=2)
print("Average L2 distance between original and original adversarial images: ", orig_l2.mean())
print("Average L2 distance between original and modified adversarial images: ", modf_l2.mean())
