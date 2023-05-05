import time
import torch
import torchattacks

from tqdm import tqdm
from torch import nn


ATTACK_MAPPINGS = {
    "deepfool": torchattacks.DeepFool,
    "cnw"     : torchattacks.CW
}

def hybridize(classifier_model, autoencoder_model, device):
    spat_classifier = nn.Sequential(
            autoencoder_model.decoder,
            classifier_model.model,
        ).to(device)

    return spat_classifier

def execute_torchattack(args, config,
                        train_dataloader,
                        autoencoder_model,
                        classifier_model,
                        attack_name, dataset_name,
                        kwargs_orig, kwargs_modf):
    # get hybrid classifier and classifier toh hai hi
    spat_classifier = hybridize(classifier_model, autoencoder_model, args.device).eval()
    autoencoder       = autoencoder_model.to(args.device).eval()
    classifier        = classifier_model.to(args.device).eval()

    # get attacks
    orig_atk = ATTACK_MAPPINGS[args.attack_name](classifier, **kwargs_orig)
    spat_atk = ATTACK_MAPPINGS[args.attack_name](spat_classifier, **kwargs_modf)

    result = {attack_name.__name__: {}}
    xs = []
    true_labels = [] # == ys

    orig_adv_images = []
    spat_adv_images = []

    orig_pred_labels = []
    spat_pred_labels = []
    for batch_idx, (data, target) in enumerate(tqdm(train_dataloader)):
        if not batch_idx:
            print("Performing attack for the first batch")
            start = time.time()

        # store before moving to device
        xs.append(data)
        true_labels.append(target)
        x, y = data.to(args.device), target.to(args.device)

        # perform original attack
        x_adv = orig_atk(x, y)
        orig_pred_label = torch.argmax(classifier(x_adv), dim=1).detach().cpu()

        # perform spat
        with torch.no_grad():
            z = autoencoder.get_z(x)
        z_adv = spat_atk(z, y)

        # get back x_hat_adv
        x_hat_adv = autoencoder.get_x_hat(z_adv)
        spat_pred_label = torch.argmax(classifier(x_hat_adv), dim=1).detach().cpu()

        if not batch_idx:
            print(f"Time taken: {time.time() - start} seconds")

        # collect all
        orig_adv_images.append(x_adv)
        spat_adv_images.append(x_hat_adv)
        orig_pred_labels.append(orig_pred_label)
        spat_pred_labels.append(spat_pred_label)
    
    # Concatenate all batches into a single tensor
    x               = torch.cat(xs, dim=0)
    true_labels     = torch.cat(true_labels, dim=0)

    orig_adv_images = torch.cat(orig_adv_images, dim=0)
    spat_adv_images = torch.cat(spat_adv_images, dim=0)
    
    orig_pred_labels = torch.cat(orig_pred_labels, dim=0)
    spat_pred_labels = torch.cat(spat_pred_labels, dim=0)

    x_adv_acc = torch.sum(true_labels == orig_pred_labels).item() / len(true_labels)
    x_hat_adv_acc = torch.sum(true_labels == spat_pred_labels).item() / len(true_labels)

    result[attack_name.__name__]["x"] = x

    result[attack_name.__name__]["x_adv"] = orig_adv_images
    result[attack_name.__name__]["x_hat_adv"] = spat_adv_images

    result[attack_name.__name__]["x_adv_acc"] = x_adv_acc
    result[attack_name.__name__]["x_hat_adv_acc"] = x_hat_adv_acc

    result[attack_name.__name__]["modf_x_adv_acc"] = 0
    result[attack_name.__name__]["modf_x_adv"] = None
    result[attack_name.__name__]["delta_x"] = None
    result[attack_name.__name__]["delta_x_hat"] = None

    return result, true_labels
 
# def deepfool_attack(model, dataloader, device='cuda', batch_size=128):
#     model = model.to(device)
#     model.eval()
#     atk = torchattacks.DeepFool(model, steps=3)

#     adv_images = []
#     true_labels = []
#     pred_labels = []

#     for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
#         data, target = data.to(device), target.to(device)
#         adv = atk(data, target)
        
#         # Collect adversarial images and corresponding labels and predictions
#         adv_images.append(adv.detach().cpu())
#         true_labels.append(target.detach().cpu())
#         pred_labels.append(torch.argmax(model(adv), dim=1).detach().cpu())

#     # Concatenate all batches into a single tensor
#     adv_images = torch.cat(adv_images, dim=0)
#     true_labels = torch.cat(true_labels, dim=0)
#     pred_labels = torch.cat(pred_labels, dim=0)

#     # Evaluate accuracy and print results
#     accuracy = torch.sum(true_labels == pred_labels).item() / len(true_labels)
#     print("Accuracy of the model on the adversarial examples: {:.2f}%".format(accuracy*100))

#     return adv_images
