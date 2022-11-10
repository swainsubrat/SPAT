from attacks import ATTACK_MAPPINGS
from attacks.plot_attack import plot_robust_accuracy

import os
import yaml
import argparse
import datetime
import warnings

from pathlib import Path
from typing import Dict

import torch
import numpy as np
from torch import nn
from attacks.art_attack import (get_models, get_xyz,
                                hybridize, execute_attack)
from dataloader import DATALOADER_MAPPINGS


def get_args_parser():
    parser = argparse.ArgumentParser('Original and modified adversarial attack', add_help=False)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--attack-name', default="fgsm", type=str,
                        help="choose one of: fgsm, bim, pgd, cnw, deepfool, jsma")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--model-name', default='mnist_ann_1', type=str,
                        help='name of the model to attack(find the list)')
    parser.add_argument('--ae-name', default='ann_128', type=str,
                        help='name of the autoencoder to use for recon(find from configs)')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='generate plot')
    parser.add_argument('--plot-dir', default='./plots', type=str,
                        help='store plot')

    return parser

def main(args):
    device = torch.device(args.device)
    attack_name = ATTACK_MAPPINGS.get(args.attack_name)
    dataset_name = args.model_name.split("_")[0]

    classifier_model, autoencoder_model, config = get_models(args)
    _, _, test_dataloader = DATALOADER_MAPPINGS[config["dataset_name"]](batch_size=args.batch_size)

    x, y, z = get_xyz(args, autoencoder_model, test_dataloader)
    
    config["latent_shape"] = args.ae_name.split('_')[-1]
    classifier, hybrid_classifier, accuracy = hybridize(x, y, z, 
                                                        config, classifier_model, autoencoder_model)

    # Perform attack
    result: Dict = execute_attack(attack_name, x, y, z, classifier, hybrid_classifier, autoencoder_model)
    print(result["FastGradientMethod"]["xx_test_adv"].shape)

    # Plot attack
    # if args.plot:
    #     is_success = plot_robust_accuracy(result, args.plot_dir, accuracy, f"robust_accuracy_{dataset_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Attack and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)