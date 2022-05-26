import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import os
import numpy as np
import foolbox as fb
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from typing import Tuple
from dataloader import load_mnist, load_cifar
from foolbox.utils import accuracy, samples
from models.classifier import MNISTClassifier, CIFAR10Classifier
from models.autoencoder import ANNAutoencoder, BaseAutoEncoder, CIFAR10Autoencoder

# first load the autoencoder and the classifier
def make_hybrid_model(autoencoder_class: BaseAutoEncoder=ANNAutoencoder,
                      classifier_class: pl.LightningModule=MNISTClassifier,
                      autoencoder_path: str="./lightning_logs/version_0/checkpoints/epoch=9-step=9370.ckpt",
                      classifier_path: str="./lightning_logs/version_6/checkpoints/epoch=9-step=9370.ckpt",
                      bounds: Tuple=(-1, 15)):
    autoencoder_model = autoencoder_class.load_from_checkpoint(autoencoder_path).to(device)
    classifier_model  = classifier_class.load_from_checkpoint(classifier_path).to(device)
    autoencoder_model.eval()
    classifier_model.eval()

    # create the hybrid model by combining decoder of the
    # autoencoder and the classifier
    hybrid_model = nn.Sequential(
        autoencoder_model.decoder,
        classifier_model.model
    )
    hybrid_model.eval()
    fmodel = fb.PyTorchModel(hybrid_model, bounds=bounds)

    return fmodel, autoencoder_model, classifier_model

fmodel, autoencoder_model, classifier_model = make_hybrid_model(CIFAR10Autoencoder, CIFAR10Classifier,
                                              autoencoder_path="./lightning_logs/version_8/checkpoints/epoch=499-step=87500.ckpt",
                                              classifier_path="./lightning_logs/version_10/checkpoints/epoch=49-step=35150.ckpt",
                                              bounds=(-5, 15))
# fetch images and format it appropiately
batch_size = 1
train_dataloader, _, _ = load_cifar(
    batch_size=batch_size, root="/home/sweta/scratch/datasets/CIFAR/"
)
images, labels = next(iter(train_dataloader))
images = images.to(device)
labels = labels.to(device)

# get the corresponding embeddings
embeddings = autoencoder_model.get_z(images)

# Only for CIFAR10
embeddings = autoencoder_model.linear(embeddings)
embeddings = embeddings.reshape(embeddings.shape[0], -1, 4, 4)

x_hat = autoencoder_model.get_x_hat(embeddings).reshape(3, 32, 32).cpu().detach().numpy()

embeddings = embeddings.cpu().detach()
embeddings = embeddings.to(device)
embeddings.requires_grad = False

# launch an attack
attack = fb.attacks.LinfPGD()
epsilons = [i/100 for i in range(0, 10, 1)]
attacks = [
        fb.attacks.FGSM(),
        fb.attacks.LinfPGD(),
        fb.attacks.LinfBasicIterativeAttack(),
        fb.attacks.LinfDeepFoolAttack(),
        # fb.attacks.L2CarliniWagnerAttack(),
        fb.attacks.LinfAdditiveUniformNoiseAttack(),
    ]

attack_names = [
    "FGSM",
    "Linf_PGD",
    "Linf_BIM",
    "Linf_DeepFool",
    # "L2_C&W",
    "Linf_AUN"
]

images   = images.reshape(3, 32, 32).cpu().detach().numpy()
attack_success = np.zeros((len(attacks), len(epsilons), len(embeddings)), dtype=np.bool)
for i, attack in enumerate(attacks):

    flag = 1
    print(f"Running {attack_names[i]} Attack....")

    # check for the folder, if not present, make one
    is_there = os.path.exists(f"./img/{attack_names[i]}/")
    if not is_there:
        os.makedirs(f"./img/{attack_names[i]}/")
    
    # carryout the attack
    noises, advs, success = attack(fmodel, embeddings, labels, epsilons=epsilons)

    print(len(advs), success.shape)
    for j, result in enumerate(success):
        if result[0].item() == True:
            fake_img = autoencoder_model.get_x_hat(advs[j])
            preds = classifier_model(fake_img)
            preds = F.log_softmax(preds, dim=-1)

            actual = labels[0].cpu().detach()
            predicted = preds.max(1, keepdim=True)[1][0][0].cpu().detach()

            print(f"{attack_names[i]} Success at Epsilon: {epsilons[j]}!!!")
            print("Actual: ", actual)
            print("Predicted: ", predicted)

            fake_img = fake_img.reshape(3, 32, 32).cpu().detach().numpy()
            diff_image = fake_img - x_hat

            plt.gray()
            fig, axis = plt.subplots(1, 4)
            axis[0].imshow(np.transpose(images, (1, 2, 0)))
            axis[0].set_title("(a) Original", loc="left")
            axis[1].imshow(np.transpose(x_hat, (1, 2, 0)))
            axis[1].set_title("(b) Reconstructed", loc="left")
            axis[2].imshow(np.transpose(fake_img, (1, 2, 0)))
            axis[2].set_title("(c) Adversaril", loc="left")
            axis[3].imshow(np.transpose(diff_image, (1, 2, 0)))
            axis[3].set_title("(d) Noises", loc="left")
            plt.savefig(f"./img/{attack_names[i]}/epsilon_{epsilons[j]}.png", dpi=600)
            break

    assert success.shape == (len(epsilons), len(embeddings))
    success_ = success.cpu().numpy()
    assert success_.dtype == np.bool
    attack_success[i] = success_
    print("  ", 1.0 - success_.mean(axis=-1).round(2))

# calculate and report the robust accuracy (the accuracy of the model when
# it is attacked) using the best attack per sample
robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
print("")
print("-" * 79)
print("")
print("worst case (best attack per-sample)")
print("  ", robust_accuracy.round(2))
print("")

print("robust accuracy for perturbations with")
for eps, acc in zip(epsilons, robust_accuracy):
    print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    # reconstruct the image from the embedding with the lowest epsilon value
    # for j, result in enumerate(success):
    #     if result[0].item() == True:
    #         fake_img = autoencoder_model.get_x_hat(advs[j])
    #         fake_img = fake_img.reshape(28, 28).cpu().detach().numpy()
    #         # preds = classifier_model(fake_img)
    #         # preds = F.log_softmax(preds, dim=-1)

    #         # actual = labels[0].cpu().detach()
    #         # predicted = preds.max(1, keepdim=True)[1][0][0].cpu().detach()

    #         print(f"{attack_names[i]} Success at Epsilon: {epsilons[j]}!!!")
    #         # print("Actual: ", actual)
    #         # print("Predicted: ", predicted)
    #         flag = 0
    #         # plt.gray()
    #         # fig, axis = plt.subplots(2)
    #         # axis[0].imshow(x_hat)
    #         # axis[1].imshow(fake_img)
    #         # plt.savefig(f"./img/{attack_names[i]}/epsilon_{epsilon}.png", dpi=600)
    #         break
    
    # if flag:
    #     print("Couldn't find any attacks :(")
