import argparse
import datetime
import os
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
from attacks.evaluate_attack import calculate_lpips
from dataloader import DATALOADER_MAPPINGS
from utils import set_logger, save, load
from attacks.torch_attacks import execute_torchattack
from attacks.plot_attack import plot_block

import warnings
warnings.filterwarnings('ignore')


def run_attack(train_dataloader, autoencoder_model, args, config, attack_name, dataset_name, classifier_model, kwargs_orig, kwargs_modf):
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
        if dataset_name == "celaba":
            config["latent_shape"] = (128, 16, 16)
        else:
            config["latent_shape"] = args.ae_name.split('_')[-1]
        classifier, hybrid_classifier, ca, ra = hybridize(x, y, z, 
                                                            config, classifier_model, autoencoder_model)
        xs.append(images)
        for ele in y[1]:
            ys.append(ele)
        cas.append(ca)
        ras.append(ra)
        # Perform attack
        conditionals = {
            "calculate_original": not args.skip_orig,
            "is_class_constrained": args.cc
        }
        results: Dict = execute_attack(config,
                                       attack_name,
                                       x, y, z,
                                       classifier,
                                       hybrid_classifier,
                                       autoencoder_model,
                                       kwargs_orig, kwargs_modf,
                                       conditionals)[attack_name.__name__]
        # results = result[attack_name.__name__]
        if not args.skip_orig:
            x_adv.append(results["x_adv"])
            x_adv_acc.append(results["x_adv_acc"])
            delta_x.append(results["delta_x"])
            orig_time.append(results["orig_time"])

        modf_x_adv.append(results["modf_x_adv"])
        modf_x_adv_acc.append(results["modf_x_adv_acc"])
        z_adv.append(results["z_adv"])
        x_hat_adv.append(results["x_hat_adv"])
        x_hat_adv_acc.append(results["x_hat_adv_acc"])
        delta_x_hat.append(results["delta_x_hat"])

        modf_time.append(results["modf_time"])

    print("Accuracy on benign test examples: {}%".format((sum(cas)/len(cas)) * 100))
    print("Accuracy on benign test examples(from reconstructed): {}%".format((sum(ras)/len(ras)) * 100))

    if not args.skip_orig:
        result[attack_name.__name__]["x_adv"] = np.vstack(x_adv)
        result[attack_name.__name__]["x_adv_acc"] = sum(x_adv_acc) / len(x_adv_acc)
        result[attack_name.__name__]["delta_x"] = np.vstack(delta_x)

    result[attack_name.__name__]["modf_x_adv"] = np.vstack(modf_x_adv)
    result[attack_name.__name__]["modf_x_adv_acc"] = sum(modf_x_adv_acc) / len(modf_x_adv_acc)
    result[attack_name.__name__]["z_adv"] = np.vstack(z_adv)
    result[attack_name.__name__]["x_hat_adv"] = np.vstack(x_hat_adv)
    result[attack_name.__name__]["x_hat_adv_acc"] = sum(x_hat_adv_acc) / len(x_hat_adv_acc)
    result[attack_name.__name__]["delta_x_hat"] = np.vstack(delta_x_hat)
    result[attack_name.__name__]["x"] = torch.vstack(xs)
    ys = np.array(ys)

    if not args.skip_orig:
        print(f"Time taken for original attack: {sum(orig_time)} seconds")
    print(f"Time taken for modified attack: {sum(modf_time)} seconds")

    return result, ys

def get_args_parser():
    parser = argparse.ArgumentParser('X and Semantic-X adversarial attack', add_help=False)
    parser.add_argument('--root-dir', default="/scratch/itee/uqsswain/",
                        help="folder where all the artifacts lie; differs from system to system")
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--dataset-len', default=1000, type=int)
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

    parser.add_argument('--eval', action='store_true', default=False,
                        help='perform attack inference')
    parser.add_argument('--skip-orig', action='store_true', default=False,
                        help='calculate original')
    parser.add_argument('--cc', action='store_true', default=False,
                        help='class-constrained')
    parser.add_argument('--torch-attacks', action='store_true', default=False,
                        help='Use torchattacks library')

    return parser

def main(args):
    logger = set_logger(args)
    dataset_name = args.model_name.split("_")[0]
    logger.info(f"Working on the dataset: {dataset_name}!!!!!")
    object_path = args.root_dir + f"artifacts/spaa/objects/{dataset_name}/{args.attack_name}.pkl"

    attack_name = ATTACK_MAPPINGS.get(args.attack_name)
    logger.info(f"----------------------------- {attack_name.__name__} ----------------------------------")

    if not args.eval:
        # Load Models
        classifier_model, autoencoder_model, config = get_models(args)
        logger.info(f"Loaded classifier and autoencoder models in eval mode!!!!!")
        train_dataloader = DATALOADER_MAPPINGS[dataset_name + "_x"](
            batch_size=args.batch_size, root=args.root_dir, dataset_len=args.dataset_len
        )
        logger.info(f"Loaded dataloader!!!!!")

        if args.torch_attacks:
            kwargs = config["torch_attack_kwargs"]
            kwargs_orig = kwargs[f"{args.attack_name}_orig"]
            kwargs_modf = kwargs[f"{args.attack_name}_modf"]
            print(kwargs_orig, kwargs_modf)
            # Run Attack torchattacks
            result, ys = execute_torchattack(args, config,
                                             train_dataloader,
                                             autoencoder_model,
                                             classifier_model,
                                             attack_name, dataset_name,
                                             kwargs_orig, kwargs_modf)
        else:
            kwargs = config["art_attack_kwargs"]
            kwargs_orig = kwargs[f"{args.attack_name}_orig"]
            kwargs_modf = kwargs[f"{args.attack_name}_modf"]
            kwargs_orig["batch_size"], kwargs_modf["batch_size"] = args.batch_size, args.batch_size
            print(kwargs_orig, kwargs_modf)
            # Run Attack ART
            result, ys = run_attack(train_dataloader,
                                    autoencoder_model,
                                    args, config,
                                    attack_name, dataset_name,
                                    classifier_model,
                                    kwargs_orig, kwargs_modf)

        # Save Adversarial Samples
        save(object_path, result)
    else:
        result = load(object_path)
    
    if not args.skip_orig:
        print("Robust accuracy of original adversarial attack: {}%".format(result[attack_name.__name__]["x_adv_acc"] * 100))
    print("Robust accuracy of modified adversarial attack: {}%".format(result[attack_name.__name__]["modf_x_adv_acc"] * 100))
    print("Robust accuracy of reconstructed adversarial attack: {}%".format(result[attack_name.__name__]["x_hat_adv_acc"] * 100))

    # # LPIPS
    x = result[attack_name.__name__]["x"]
    orig_lpips, modf_lpips, orig_linf, modf_linf = calculate_lpips(args, result, attack_name, x)
    print("Average LPIPS score of original adversarial attack: ", orig_lpips.flatten().mean())
    print("Average LPIPS score of modifed adversarial attack: ", modf_lpips.flatten().mean())

    # Linf
    print("Average Linf distance between original and original adversarial images: ", orig_linf.mean())
    print("Average Linf distance between original and modified adversarial images: ", modf_linf.mean())

    if args.plot:
        # Plot
        if not args.skip_orig:
            x_adv = result[attack_name.__name__]["x_adv"]
            delta_x = result[attack_name.__name__]["delta_x"]
        
        x_hat_adv  = result[attack_name.__name__]["x_hat_adv"]
        modf_x_adv = result[attack_name.__name__]["modf_x_adv"]

        # noises
        delta_x_hat = result[attack_name.__name__]["delta_x_hat"]

        start = 22
        end = 32
        images1 = np.vstack([x[start: end], x_adv[start: end], delta_x[start: end]])
        images2 = np.vstack([modf_x_adv[start: end], delta_x_hat[start: end]])
        plot_block(images1, "./plots/cifar_df_block1.png")
        plot_block(images2, "./plots/cifar_df_block2.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Attack and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)