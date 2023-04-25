import argparse
import datetime
import os
import warnings
from pathlib import Path
from typing import Dict

import logging
import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from attacks import ATTACK_MAPPINGS
from attacks.art_attack import execute_attack, get_models, get_xyz, hybridize
from attacks.plot_attack import plot_lips, plot_robust_accuracy
from dataloader import DATALOADER_MAPPINGS
from utils import set_logger


def run_attack(train_dataloader, autoencoder_model, args, config, attack_name, dataset_name, classifier_model):
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

        if dataset_name == "imagenet":
            config["latent_shape"] = (512, 7, 7)
        else:
            config["latent_shape"] = (args.ae_name.split('_')[-1], )
        kwargs = {"norm": "inf", "eps": 0.01}
        classifier, hybrid_classifier, ca, ra = hybridize(x, y, z, 
                                                            config, classifier_model, autoencoder_model)
        xs.append(x[0])
        for ele in y[1]:
            ys.append(ele)
        cas.append(ca)
        ras.append(ra)
        # Perform attack
        conditionals = {
            "calculate_original": True,
            "is_class_constrained": False
        }
        results: Dict = execute_attack(config, attack_name, x, y, z, classifier, hybrid_classifier, autoencoder_model, kwargs, conditionals)[attack_name.__name__]
        # results = result[attack_name.__name__]
        x_adv.append(results["x_adv"])
        x_adv_acc.append(results["x_adv_acc"])
        delta_x.append(results["delta_x"])
        modf_x_adv.append(results["modf_x_adv"])
        modf_x_adv_acc.append(results["modf_x_adv_acc"])
        z_adv.append(results["z_adv"])
        x_hat_adv.append(results["x_hat_adv"])
        x_hat_adv_acc.append(results["x_hat_adv_acc"])
        delta_x_hat.append(results["delta_x_hat"])

        orig_time.append(results["orig_time"])
        modf_time.append(results["modf_time"])

    print("Accuracy on benign test examples: {}%".format((sum(cas)/len(cas)) * 100))
    print("Accuracy on benign test examples(from reconstructed): {}%".format((sum(ras)/len(ras)) * 100))

    result[attack_name.__name__]["x_adv"] = np.vstack(x_adv)
    result[attack_name.__name__]["x_adv_acc"] = sum(x_adv_acc) / len(x_adv_acc)
    result[attack_name.__name__]["delta_x"] = np.vstack(delta_x)

    result[attack_name.__name__]["modf_x_adv"] = np.vstack(modf_x_adv)
    result[attack_name.__name__]["modf_x_adv_acc"] = sum(modf_x_adv_acc) / len(modf_x_adv_acc)
    result[attack_name.__name__]["z_adv"] = np.vstack(z_adv)
    result[attack_name.__name__]["x_hat_adv"] = np.vstack(x_hat_adv)
    result[attack_name.__name__]["x_hat_adv_acc"] = sum(x_hat_adv_acc) / len(x_hat_adv_acc)
    result[attack_name.__name__]["delta_x_hat"] = np.vstack(delta_x_hat)
    xs = torch.vstack(xs)
    ys = np.array(ys)

    print(f"Time taken for original attack: {sum(orig_time)} seconds")
    print(f"Time taken for modified attack: {sum(modf_time)} seconds")

    return result

def get_args_parser():
    parser = argparse.ArgumentParser('X and Semantic-X adversarial attack', add_help=False)
    parser.add_argument('--root-dir', default="/scratch/itee/uqsswain/",
                        help="folder where all the artifacts lie; differs from system to system")
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--attack-name', default="fgsm", type=str,
                        help="choose one of: fgsm, bim, pgd, cnw, deepfool, elastic")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--model-name', default='mnist_ann_1', type=str,
                        help='name of the model to attack(find the list)')
    parser.add_argument('--ae-name', default='ann_128', type=str,
                        help='name of the autoencoder to use for recon(find from configs)')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='generate plot')

    return parser

def main(args):
    #TODO: change the file mode in logger from "w" to "a" later
    logger = set_logger(args)
    attack_name = ATTACK_MAPPINGS.get(args.attack_name)
    dataset_name = args.model_name.split("_")[0]
    logger.info(f"Working on the dataset: {dataset_name}!!!!!")
    logger.info(f"----------------------------- {attack_name.__name__} ----------------------------------")

    classifier_model, autoencoder_model, config = get_models(args)
    logger.info(f"Loaded classifier and autoencoder models in eval mode!!!!!")
    train_dataloader = DATALOADER_MAPPINGS[dataset_name + "_x"](batch_size=args.batch_size, root=args.root_dir)
    logger.info(f"Loaded dataloader!!!!!")

    # x, y, z = get_xyz(args, autoencoder_model, train_dataloader)
    result = run_attack(train_dataloader, autoencoder_model, args, config, attack_name, dataset_name, classifier_model)

    print("Robust accuracy of original adversarial attack: {}%".format(result[attack_name.__name__]["x_adv_acc"] * 100))
    print("Robust accuracy of modified adversarial attack: {}%".format(result[attack_name.__name__]["modf_x_adv_acc"] * 100))
    print("Robust accuracy of reconstructed adversarial attack: {}%".format(result[attack_name.__name__]["x_hat_adv_acc"] * 100))

    # config["latent_shape"] = args.ae_name.split('_')[-1]
    # classifier, hybrid_classifier, accuracy = hybridize(x, y, z, 
    #                                                     config, classifier_model, autoencoder_model)

    # # Perform attack
    # result: Dict = execute_attack(config, attack_name, x, y, z, classifier, hybrid_classifier, autoencoder_model)
    # # plot_adversarial_images(result, args.plot_dir, "temp")

    # # Plot attack
    # if args.plot:
    #     is_success = plot_robust_accuracy(result, args.plot_dir, accuracy, f"robust_accuracy_{dataset_name}_hybrid")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Attack and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)