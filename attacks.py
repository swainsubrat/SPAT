import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import os
import numpy as np
import foolbox as fb
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from typing import Tuple, Callable
from dataloader import load_mnist, load_cifar, load_fashion_mnist
from foolbox.utils import accuracy, samples
from models.classifier import MNISTClassifier, CIFAR10Classifier
from models.autoencoder import ANNAutoencoder, BaseAutoEncoder, CIFAR10Autoencoder


def make_hybrid_model(autoencoder_class: BaseAutoEncoder=ANNAutoencoder,
                      classifier_class: pl.LightningModule=MNISTClassifier,
                      autoencoder_path: str="./lightning_logs/version_0/checkpoints/epoch=9-step=9370.ckpt",
                      classifier_path: str="./lightning_logs/version_6/checkpoints/epoch=9-step=9370.ckpt",
                      bounds: Tuple=(-1, 15)):
    """
    A function to load individual models(autoencoder and classifier) and
    make a hybrid model.
    """
    # first load the autoencoder and the classifier
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

def get_embeddings(root: str=None,
                   load_function: Callable=load_mnist,
                   batch_size: int=100):
    """
    passing the input to the encoder and getting the corresponding
    embedding.
    """
    # fetch images and format it appropiately
    train_dataloader, _, _ = load_function(
        batch_size=batch_size, root=root
    )
    images, labels = next(iter(train_dataloader))
    images = images.to(device)
    labels = labels.to(device)
    
    # get the corresponding embeddings
    embeddings = autoencoder_model.get_z(images)
    
    # Only for CIFAR10
    # embeddings = autoencoder_model.linear(embeddings)
    # embeddings = embeddings.reshape(embeddings.shape[0], -1, 4, 4)
    
    if PLOT:
        x_hat = autoencoder_model.get_x_hat(embeddings).reshape(RESHAPE).cpu().detach().numpy()
    else:
        x_hat = None
    
    embeddings = embeddings.cpu().detach()
    embeddings = embeddings.to(device)
    embeddings.requires_grad = False
    
    return embeddings, images, labels, x_hat

# constants
ROOT             = "/home/sweta/scratch/datasets/MNIST/"
AUTOENCODER_PATH = "./lightning_logs/version_0/checkpoints/epoch=9-step=9370.ckpt"
CLASSIFIER_PATH  = "./lightning_logs/version_6/checkpoints/epoch=9-step=9370.ckpt"
BOUNDS           = (-5, 15)
PLOT             = True
BATCH_SIZE       = 1 if PLOT else 100
RESHAPE          = (28, 28)
LOAD_FUNCTION    = load_mnist

epsilons = [i/100 for i in range(1, 100, 5)]
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

fmodel, autoencoder_model, classifier_model = make_hybrid_model(
                                              autoencoder_path=AUTOENCODER_PATH,
                                              classifier_path=CLASSIFIER_PATH,
                                              bounds=BOUNDS)
embeddings, images, labels, x_hat = get_embeddings(root=ROOT, load_function=LOAD_FUNCTION, batch_size=BATCH_SIZE)

if PLOT:
    images = images.reshape(RESHAPE).cpu().detach().numpy()

# launch an attack
attack_success = np.zeros((len(attacks), len(epsilons), len(embeddings)), dtype=np.bool)
for i, attack in enumerate(attacks):

    print(f"Running {attack_names[i]} Attack....")

    # carryout the attack
    noises, advs, success = attack(fmodel, embeddings, labels, epsilons=epsilons)

    assert success.shape == (len(epsilons), len(embeddings))
    success_ = success.cpu().numpy()
    assert success_.dtype == np.bool
    attack_success[i] = success_
    print("  ", 1.0 - success_.mean(axis=-1).round(2))

    """
    code for the plot
    """
    if PLOT:
        # check for the folder, if not present, make one
        is_there = os.path.exists(f"./img/{attack_names[i]}/")
        if not is_there:
            os.makedirs(f"./img/{attack_names[i]}/")
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

                fake_img = fake_img.reshape(RESHAPE).cpu().detach().numpy()
                diff_image = fake_img - images

                plt.gray()
                fig, axis = plt.subplots(1, 4)
                fig.suptitle(f"Acutal: {actual} but Predicted: {predicted}")
                # axis[0].imshow(np.transpose(images, (1, 2, 0)))
                axis[0].imshow(images)
                axis[0].set_title("(a) Original", loc="left")
                # axis[1].imshow(np.transpose(x_hat, (1, 2, 0)))
                axis[1].imshow(x_hat)
                axis[1].set_title("(b) Recons", loc="left")
                # axis[2].imshow(np.transpose(fake_img, (1, 2, 0)))
                axis[2].imshow(fake_img)
                axis[2].set_title("(c) Adversarial", loc="left")
                # axis[3].imshow(np.transpose(diff_image, (1, 2, 0)))
                axis[3].imshow(diff_image)
                axis[3].set_title("(d) Noises", loc="left")
                plt.savefig(f"./img/{attack_names[i]}/epsilon_{epsilons[j]}.png", dpi=600)
                break

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
