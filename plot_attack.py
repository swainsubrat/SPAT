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
from utils import save, load
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
        # fb.attacks.FGSM(),
        # fb.attacks.LinfPGD(),
        # fb.attacks.LinfBasicIterativeAttack(),
        fb.attacks.LinfDeepFoolAttack(),
        # fb.attacks.L2CarliniWagnerAttack(),
        # fb.attacks.LinfAdditiveUniformNoiseAttack(),
    ]

attack_names = [
    # "FGSM",
    # "Linf_PGD",
    # "Linf_BIM",
    "Linf_DeepFool",
    # "L2_C&W",
    # "Linf_AUN"
]

def filter_images(images: Tensor, labels: Tensor):
    visited = [False] * 10
    ten_images, ten_lables = [None]*10, [None]*10
    for img, lbl in zip(images, labels):
        if not visited[lbl.item()]:
            ten_images[lbl.item()] = img
            ten_lables[lbl.item()] = lbl
            visited[lbl.item()] = True
    ten_images = torch.stack(ten_images)
    ten_lables = torch.stack(ten_lables)

    return ten_images, ten_lables

# fetch images and format it appropiately
# _, _, test_dataloader = LOAD_FUNCTION(
#     batch_size=BATCH_SIZE, root=ROOT
# )
# images, labels = next(iter(test_dataloader))
# ten_images, ten_labels = filter_images(images=images, labels=labels)

# save("objects/ten.pkl", {"images": ten_images, "labels": ten_labels})
saved_tens = load("objects/ten.pkl")

ten_images, ten_labels = saved_tens["images"], saved_tens["labels"]

# call appropiate methods to get models and embeddings
fmodel, orig_fmodel, autoencoder_model, classifier_model = make_hybrid_model(
                                            autoencoder_path=AUTOENCODER_PATH,
                                            classifier_path=CLASSIFIER_PATH,
                                            bounds=BOUNDS)
embeddings, images, labels, x_hat = get_embeddings(autoencoder_model=autoencoder_model,
                                                    images=ten_images, labels=ten_labels)

# launch an attack
for i, attack in enumerate(attacks):

    print(f"Running {attack_names[i]} Attack....")

    # carryout the attack on original
    start = time.time()
    orig_noises, orig_advs, orig_success = attack(orig_fmodel, images, labels, epsilons=epsilons)
    print(f"Time spent on original attack: {time.time() - start}")

    # carryout the modified attack
    start = time.time()
    noises, advs, success = attack(fmodel, embeddings, labels, epsilons=epsilons)
    print(f"Time spent on modified attack: {time.time() - start}")

plt.figure(figsize=(18, 4))
INDEX = -1
fake_images = []
orig_diff_img = images - orig_noises[INDEX]
# for i, img in enumerate(images):
#     ylabel = "Original" if not i else None
#     img = img.reshape(RESHAPE).cpu()
#     plt.subplot(5, 10, i+1, xticks=[], yticks=[], ylabel=ylabel)
#     plt.imshow(img)
# for i, img in enumerate(orig_noises[INDEX]):
#     label = np.argmax(classifier_model(img).detach().cpu()).item()
#     ylabel = attack_names[0] if not i else None
#     img = img.reshape(RESHAPE).cpu()
#     plt.subplot(5, 10, i+11, xticks=[], yticks=[], ylabel=ylabel, xlabel=f"Predicted: {label}")
#     plt.imshow(img)
# for i, img in enumerate(orig_diff_img):
#     ylabel = "Noise" if not i else None
#     img = img.reshape(RESHAPE).cpu()
#     plt.subplot(5, 10, i+21, xticks=[], yticks=[], ylabel=ylabel)
#     plt.imshow(img)
for i, adv in enumerate(noises[INDEX]):
    ylabel = f"{attack_names[0]}" if not i else None
    img = autoencoder_model.get_x_hat(adv)
    label = np.argmax(classifier_model(img).detach().cpu()).item()
    fake_images.append(img)
    img = img.reshape(RESHAPE).detach().cpu()
    # plt.subplot(5, 10, i+31, xticks=[], yticks=[], ylabel=ylabel, xlabel=f"Predicted: {label}")
    # plt.imshow(img)
fake_images = torch.stack(fake_images)
diff_img = images - fake_images

# import pdb; pdb.set_trace()
orig_norm_inf = np.linalg.norm(orig_diff_img.detach().cpu(), ord=np.inf, axis=1)
norm_inf      = np.linalg.norm(diff_img.detach().cpu(), ord=np.inf, axis=1)
orig_norm_2 = np.linalg.norm(orig_diff_img.detach().cpu(), ord=2, axis=1)
norm_2      = np.linalg.norm(diff_img.detach().cpu(), ord=2, axis=1)

print(f"Linf Norm of Original Attacks: {orig_norm_inf}")
print(f"Linf Norm of Modified Attacks: {norm_inf}")
print(f"L2 Norm of Original Attacks: {orig_norm_2}")
print(f"L2 Norm of Original Attacks: {norm_2}")

for i, linf_norm in enumerate(orig_norm_inf):
    ylabel = "Deepfool" if not i else None
    img = orig_diff_img[i].reshape(RESHAPE).detach().cpu()
    plt.subplot(2, 10, i+1, xticks=[], yticks=[], ylabel=ylabel, xlabel=f"Linf: {orig_norm_inf[i]:4.1f}, L2: {orig_norm_2[i]:4.1f}")
    plt.imshow(img)
    
for i, linf_norm in enumerate(norm_inf):
    ylabel = "Our Method" if not i else None
    img = diff_img[i].reshape(RESHAPE).detach().cpu()
    plt.subplot(2, 10, i+11, xticks=[], yticks=[], ylabel=ylabel, xlabel=f"Linf: {norm_inf[i]:4.1f}, L2: {norm_2[i]:4.1f}")
    plt.imshow(img)
plt.savefig('img/noise_norm_deepfool.jpg')
# for i, img in enumerate(diff_img):
#     ylabel = "Noise" if not i else None
#     img = img.reshape(RESHAPE).detach().cpu()
#     plt.subplot(5, 10, i+41, xticks=[], yticks=[], ylabel=ylabel)
#     plt.imshow(img)
# plt.savefig('img/orig_attack_aun.jpg')

# print(type(noises), type(advs), type(success))
# print(noises[-1].shape, advs[-1].shape)
# print(noises[-1][0].shape)