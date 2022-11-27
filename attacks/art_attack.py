import sys

import yaml

sys.path.append("..")

import numpy as np
import pytorch_lightning as pl
import torch
from art.estimators.classification import PyTorchClassifier
from torch import nn

from attacks import ATTACK_MAPPINGS
from models import MODEL_MAPPINGS


def get_xyz(args, autoencoder_model, test_dataloader):
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

    config["device"]       = torch.device(args.device)
    config["dataset_name"] = dataset_name

    classifier_path = config["classifiers"][args.model_name]
    autoencoder_path = config["autoencoders"][args.ae_name]

    classifier_model_class = MODEL_MAPPINGS[classifier_path]
    autoencoder_model_class = MODEL_MAPPINGS[autoencoder_path]

    classifier_model = classifier_model_class.load_from_checkpoint(classifier_path).to(args.device)
    autoencoder_model = autoencoder_model_class.load_from_checkpoint(autoencoder_path).to(args.device)

    classifier_model.eval()
    autoencoder_model.eval()

    return classifier_model, autoencoder_model, config

def hybridize(x, y, z, config, classifier_model, autoencoder_model):
    hybrid_classifier_model = nn.Sequential(
            autoencoder_model.decoder,
            classifier_model.model
        )
    miscs = config["miscs"]
    criterion = nn.CrossEntropyLoss()
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
        input_shape=(1, config["latent_shape"]),
        nb_classes=miscs["nb_classes"],
    )

    # Step 5: Evaluate the ART classifier on benign test examples
    print(type(x[1]), x[1].shape)
    predictions = classifier.predict(x[1])
    accuracy = np.sum(np.argmax(predictions, -1) == y[1])/ len(y[1])
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    print(z[1].shape)
    # from torchsummary import summary
    # summary(hybrid_classifier_model, (128, 16, 16))
    predictions = hybrid_classifier.predict(z[1])
    accuracy = np.sum(np.argmax(predictions, -1) == y[1])/ len(y[1])
    print("Accuracy on benign test examples(from reconstructed): {}%".format(accuracy * 100))

    return classifier, hybrid_classifier, accuracy

def execute_attack(config, attack_name, x, y, z, classifier, hybrid_classifier, autoencoder_model):
    result = {}
    if attack_name == "all":
        del ATTACK_MAPPINGS["all"]
        for name, attack_name in ATTACK_MAPPINGS.items():
            result[name] = {}
            print(f"Running {name} attack!!!!!")
            attack = attack_name(classifier)
            x_test_adv_np = attack.generate(x=x[1])
            predictions = classifier.predict(x_test_adv_np)
            accuracy = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])
            result[name]["original"] = accuracy
            result[name]["x_test_adv_np"] = x_test_adv_np

            # calculate noise
            x_test_noise = x_test_adv_np - x[1]
            result[name]["x_test_noise"] = x_test_noise
            print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

            modified_attack = attack_name(hybrid_classifier)
            z_test_adv_np = modified_attack.generate(x=z[1])

            # Step 7: Evaluate the ART classifier on adversarial test examples
            predictions = hybrid_classifier.predict(z_test_adv_np)
            accuracy = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])
            result[name]["modified"] = accuracy
            result[name]["z_test_adv_np"] = z_test_adv_np
            xx_test_adv = autoencoder_model.decoder(torch.Tensor(z_test_adv_np).to(config["device"]))
            xx_test     = autoencoder_model.decoder(torch.Tensor(z[1]).to(config["device"]))

            # calculate noise
            xx_test_noise = xx_test_adv - xx_test
            result[name]["xx_test_noise"] = xx_test_noise.cpu().detach().numpy()
            result[name]["xx_test_adv_np"] = xx_test_adv.cpu().detach().numpy()
            print("Accuracy on adversarial test examples(Modified): {}%".format(accuracy * 100))

            hybrid_noise = result[name]["xx_test_noise"] + 0.1 * x_test_noise
            hybrid_x_np = x[1] + hybrid_noise
            result[name]["hybrid_x_np"] = hybrid_x_np
            result[name]["hybrid_noise"] = hybrid_noise

            predictions = classifier.predict(hybrid_x_np)
            accuracy = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])
            print("Accuracy on adversarial test examples(Hybrid): {}%".format(accuracy * 100))
            result[name]["hybrid"] = accuracy
    else:
        name = attack_name.__name__
        result[name] = {}
        attack = attack_name(classifier)
        x_test_adv_np = attack.generate(x=x[1])
        predictions = classifier.predict(x_test_adv_np)
        accuracy = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])
        result[name]["original"] = accuracy
        result[name]["x_test_adv_np"] = x_test_adv_np

        # calculate noise
        x_test_noise = x_test_adv_np - x[1]
        result[name]["x_test_noise"] = x_test_noise
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

        modified_attack = attack_name(hybrid_classifier)
        z_test_adv_np = modified_attack.generate(x=z[1])

        # Step 7: Evaluate the ART classifier on adversarial test examples
        predictions = hybrid_classifier.predict(z_test_adv_np)
        accuracy = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])
        result[name]["modified"] = accuracy
        result[name]["z_test_adv_np"] = z_test_adv_np
        xx_test_adv = autoencoder_model.decoder(torch.Tensor(z_test_adv_np).to(config["device"]))
        xx_test     = autoencoder_model.decoder(torch.Tensor(z[1]).to(config["device"]))

        # calculate noise
        xx_test_noise = xx_test_adv - xx_test
        result[name]["xx_test_noise"] = xx_test_noise.cpu().detach().numpy()
        result[name]["xx_test_adv_np"] = xx_test_adv.cpu().detach().numpy()
        print("Accuracy on adversarial test examples(Modified): {}%".format(accuracy * 100))

        hybrid_noise = result[name]["xx_test_noise"] + 0.1 * x_test_noise
        hybrid_x_np = x[1] + hybrid_noise
        result[name]["hybrid_x_np"] = hybrid_x_np
        result[name]["hybrid_noise"] = hybrid_noise

        predictions = classifier.predict(hybrid_x_np)
        accuracy = np.sum(np.argmax(predictions, axis=-1) == y[1]) / len(y[1])
        print("Accuracy on adversarial test examples(Hybrid): {}%".format(accuracy * 100))
        result[name]["hybrid"] = accuracy

    return result