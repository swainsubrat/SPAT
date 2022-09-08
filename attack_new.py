"""
File to create attack on the latent space and input space and
compare them
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

from torch import nn, Tensor
from typing import Tuple, Callable
from dataloader import load_celeba, load_mnist
from models.classifier import (MNISTClassifier, CIFAR10Classifier,
                                CelebAClassifier)
from models.autoencoder import (ANNAutoencoder, BaseAutoEncoder,
                                CIFAR10Autoencoder, CelebAAutoencoder)

# first load the config
with open("./configs/celeba.yml", "r") as f:
    config = yaml.safe_load(f)

ROOT             = config["paths"]["root"]
AUTOENCODER_PATH = config["paths"]["autoencoder-path"]
CLASSIFIER_PATH  = config["paths"]["classifier-path"]
BOUNDS           = (config["specs"]["bounds"][0], config["specs"]["bounds"][1])
PLOT             = config["specs"]["plot"]
BATCH_SIZE       = config["specs"]["batch_size"]
RESHAPE          = (config["specs"]["reshape"][0], config["specs"]["reshape"][1])
LOAD_FUNCTION    = load_celeba

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
    # print(hybrid_model)
    # import pdb; pdb.set_trace()
    hybrid_model.eval()
    fmodel = fb.PyTorchModel(hybrid_model, bounds=bounds)
    orig_fmodel = fb.PyTorchModel(classifier_model, bounds=bounds)

    return fmodel, orig_fmodel, autoencoder_model, classifier_model

def get_embeddings(autoencoder_model: BaseAutoEncoder,
                images: Tensor=None,
                labels: Tensor=None):
    """
    passing the input to the encoder and getting the corresponding
    embedding.
    """
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

if __name__ == "__main__":
    epsilons = [i/100 for i in range(1, 50, 5)]
    attacks = [
            fb.attacks.FGSM(),
            fb.attacks.LinfPGD(),
            fb.attacks.LinfBasicIterativeAttack(),
            fb.attacks.LinfDeepFoolAttack(),
            # fb.attacks.L2CarliniWagnerAttack(),
            # fb.attacks.LinfAdditiveUniformNoiseAttack(),
        ]

    attack_names = [
        "FGSM",
        "Linf_PGD",
        "Linf_BIM",
        "Linf_DeepFool",
        # "L2_C&W",
        # "Linf_AUN"
    ]

    # fetch images and format it appropiately
    _, test_dataloader = LOAD_FUNCTION(
        batch_size=BATCH_SIZE, root=ROOT
    )
    images, labels = next(iter(test_dataloader))

    # call appropiate methods to get models and embeddings
    fmodel, orig_fmodel, autoencoder_model, classifier_model = make_hybrid_model(
                                                autoencoder_class=CelebAAutoencoder,
                                                classifier_class=CelebAClassifier,
                                                autoencoder_path=AUTOENCODER_PATH,
                                                classifier_path=CLASSIFIER_PATH,
                                                bounds=BOUNDS)
    embeddings, images, labels, x_hat = get_embeddings(autoencoder_model=autoencoder_model,
                                                    images=images, labels=labels)

    if PLOT:
        images = images.reshape(RESHAPE).cpu().detach().numpy()

    # launch an attack
    import pdb; pdb.set_trace()
    orig_attack_success = np.zeros((len(attacks), len(epsilons), len(embeddings)), dtype=bool)
    attack_success = np.zeros((len(attacks), len(epsilons), len(embeddings)), dtype=bool)
    for i, attack in enumerate(attacks):

        print(f"Running {attack_names[i]} Attack....")

        # # carryout the attack on original
        # start = time.time()
        # orig_noises, orig_advs, orig_success = attack(orig_fmodel, images, labels, epsilons=epsilons)
        # print(f"Time spent on original attack: {time.time() - start}")
        # orig_attack_success[i] = orig_success.cpu().numpy()

        # carryout the modified attack
        start = time.time()
        noises, advs, success = attack(fmodel, embeddings, labels, epsilons=epsilons)
        print(f"Time spent on modified attack: {time.time() - start}")

        assert success.shape == (len(epsilons), len(embeddings))
        print(f"success: {success.shape}")
        success_ = success.cpu().numpy()
        assert success_.dtype == bool
        attack_success[i] = success_
        print("  ", 1.0 - success_.mean(axis=-1).round(2))
        print(f"attack success: {attack_success.shape}")

    # calculate and report the robust accuracy
    # orig_robust_accuracy = 1.0 - orig_attack_success.mean(axis=-1)
    robust_accuracy = 1.0 - attack_success.mean(axis=-1)

    print("original and robust accuracy for perturbations with")
    for i in range(len(epsilons)):
        print(f"Linf norm $\leq$ {epsilons[i]:<6}", end= " & ")
        for j in range(len(attack_names)):
            # print(f"{orig_robust_accuracy[j][i].item() * 100:4.1f}", end=" & ")
            print(f"{robust_accuracy[j][i].item() * 100:4.1f}", end=" & ")
        
        print("\\((")
        print("\hline")

    # print("robust accuracy for perturbations with")
    # for i in range(len(epsilons)):
    #     print(f"Linf norm ≤ {epsilons[i]:<6}", end= " & ")
    #     for j in range(len(attack_names)):
    #         print(f"{robust_accuracy[j][i].item() * 100:4.1f}", end=" & ")
        
    #     print()

# """
# code for the plot
# """
# if PLOT:
#     # check for the folder, if not present, make one
#     is_there = os.path.exists(f"./img/{attack_names[i]}/")
#     if not is_there:
#         os.makedirs(f"./img/{attack_names[i]}/")
#     for j, result in enumerate(success):
#         if result[0].item() == True:
#             fake_img = autoencoder_model.get_x_hat(advs[j])
#             preds = classifier_model(fake_img)
#             preds = F.log_softmax(preds, dim=-1)

#             actual = labels[0].cpu().detach()
#             predicted = preds.max(1, keepdim=True)[1][0][0].cpu().detach()

#             print(f"{attack_names[i]} Success at Epsilon: {epsilons[j]}!!!")
#             print("Actual: ", actual)
#             print("Predicted: ", predicted)

#             fake_img = fake_img.reshape(RESHAPE).cpu().detach().numpy()
#             diff_image = fake_img - images

#             plt.gray()
#             fig, axis = plt.subplots(1, 4)
#             fig.suptitle(f"Acutal: {actual} but Predicted: {predicted}")
#             # axis[0].imshow(np.transpose(images, (1, 2, 0)))
#             axis[0].imshow(images)
#             axis[0].set_title("(a) Original", loc="left")
#             # axis[1].imshow(np.transpose(x_hat, (1, 2, 0)))
#             axis[1].imshow(x_hat)
#             axis[1].set_title("(b) Recons", loc="left")
#             # axis[2].imshow(np.transpose(fake_img, (1, 2, 0)))
#             axis[2].imshow(fake_img)
#             axis[2].set_title("(c) Adversarial", loc="left")
#             # axis[3].imshow(np.transpose(diff_image, (1, 2, 0)))
#             axis[3].imshow(diff_image)
#             axis[3].set_title("(d) Noises", loc="left")
#             plt.savefig(f"./img/{attack_names[i]}/epsilon_{epsilons[j]}.png", dpi=600)
#             break
