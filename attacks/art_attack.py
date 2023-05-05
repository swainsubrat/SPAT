import sys
import time

import yaml

sys.path.append("..")

import numpy as np
import pytorch_lightning as pl
import torch
from art.estimators.classification import PyTorchClassifier
from torch import nn

from attacks import ATTACK_MAPPINGS, generate_mask
from models import MODEL_MAPPINGS

class hybrid_model(nn.Module):
    def __init__(self, config, autoencoder_model, classifier_model):
        super().__init__()
        self.model = nn.Sequential(
            autoencoder_model.decoder,
            classifier_model.model,
        ).to(config["device"])

    def forward(self, x):
        x = self.model(x)
        return x.flatten()

def get_xyz(args, autoencoder_model, test_dataloader):
    """
    get x, y, and z value for a particular item itered from
    the dataloader.
    """
    x_test, y_test = next(iter(test_dataloader))
    x_test, y_test = x_test.to(args.device), y_test.to(args.device)
    x_test_np, y_test_np = x_test.cpu().numpy(), y_test.cpu().numpy()

    z_test = autoencoder_model.get_z(x_test)
    z_test_np = z_test.detach().cpu().numpy()

    return (x_test, x_test_np), (y_test, y_test_np), (z_test, z_test_np)

def get_models(args):
    """
    load the corresponding model and return in the
    eval mode.
    """
    dataset_name = args.model_name.split("_")[0]
    with open(f"./configs/{dataset_name}.yml", "r") as f:
        config = yaml.safe_load(f)

    config["device"] = torch.device(args.device)
    config["dataset_name"] = dataset_name
    # if args.dataset_len:
    #     config["dataset_name"] = dataset_name + str(args.dataset_len)

    classifier_path = config["classifiers"][args.model_name]
    autoencoder_path = config["autoencoders"][args.ae_name]

    classifier_model_class = MODEL_MAPPINGS[classifier_path]
    autoencoder_model_class = MODEL_MAPPINGS[autoencoder_path]

    if dataset_name == "imagenet":
        classifier_model = classifier_model_class()
        autoencoder_model = autoencoder_model_class(configs=[2, 2, 3, 3, 3])
        checkpoint = torch.load(autoencoder_path)
        new_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            if key.startswith('module.'):
                new_key = key[7:] # remove "module." prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        autoencoder_model.load_state_dict(new_state_dict)
        autoencoder_model = autoencoder_model.to(args.device)
    else:
        classifier_model = classifier_model_class.load_from_checkpoint(classifier_path).to(args.device)
        autoencoder_model = autoencoder_model_class.load_from_checkpoint(autoencoder_path).to(args.device)

    classifier_model.eval()
    # autoencoder_model.eval()

    return classifier_model, autoencoder_model, config

def hybridize(x, y, z, config, classifier_model, autoencoder_model):
    if config["dataset_name"] == "celeba":
        hybrid_classifier_model = hybrid_model(config, autoencoder_model, classifier_model)
        criterion = nn.BCELoss()
    else:
        hybrid_classifier_model = nn.Sequential(
                autoencoder_model.decoder,
                classifier_model.model,
            ).to(config["device"])
        criterion = nn.CrossEntropyLoss()

    miscs = config["miscs"]
    # optimizer = optim.Adam(mnist_classifier.parameters(), lr=0.01)

    # Step 4: Create the ART classifier
    classifier = PyTorchClassifier( 
        model=classifier_model,
        clip_values=(miscs["min_pixel_value"], miscs["max_pixel_value"]),
        loss=criterion,
        # optimizer=optimizer,
        input_shape=tuple(miscs["input_shape"]),
        nb_classes=miscs["nb_classes"],
    )
    hybrid_classifier = PyTorchClassifier(
        model=hybrid_classifier_model,
        # clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        # optimizer=optimizer,
        input_shape=(1,) + config["latent_shape"],
        nb_classes=miscs["nb_classes"],
    )

    # Step 5: Evaluate the ART classifier on benign test examples
    predictions = classifier.predict(x[1])
    accuracy = np.sum(np.argmax(predictions, -1) == y[1])/ len(y[1])

    predictions = hybrid_classifier.predict(z[1])
    r_accuracy = np.sum(np.argmax(predictions, -1) == y[1])/ len(y[1])

    return classifier, hybrid_classifier, accuracy, r_accuracy

def execute_attack(config, attack_name, x, y, z, classifier, 
                   hybrid_classifier, autoencoder_model, kwargs_orig, kwargs_modf,
                   conditionals):
    result = {}
    name = attack_name.__name__
    result[name] = {}
    # import pdb; pdb.set_trace()
    
    # ------------------------------------------------- #
    # ---------------- Original Attack ---------------- #
    # ------------------------------------------------- #
    if conditionals["calculate_original"]:
        attack = attack_name(classifier, **kwargs_orig)
        start = time.time()
        x_adv = attack.generate(x=x[1], y=y[1])
        orig_time = time.time() - start
        predictions = classifier.predict(x_adv)
        x_adv_acc = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])

        result[name]["x_adv"] = x_adv
        result[name]["x_adv_acc"] = x_adv_acc

        # calculate noise
        delta_x = x_adv - x[1]
        result[name]["delta_x"] = delta_x


    # ------------------------------------------------- #
    # ---------------- Modified Attack ---------------- #
    # ------------------------------------------------- #
    modified_attack = attack_name(hybrid_classifier, **kwargs_modf)
    if conditionals["is_class_constrained"]:
        start = time.time()
        z_adv = modified_attack.generate(x=z[1], y=y[1], mask=generate_mask(
            latent_dim=int(config["latent_shape"]),
            n_classes=config["miscs"]["nb_classes"],
            labels=y[1]))
        modf_time = time.time() - start
    else:
        start = time.time()
        z_adv = modified_attack.generate(x=z[1], y=y[1])
        modf_time = time.time() - start

    # calculate noise
    autoencoder_model = autoencoder_model.to(config["device"])
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

    if conditionals["calculate_original"]:
        result[name]["orig_time"] = orig_time
    result[name]["modf_time"] = modf_time

    return result
