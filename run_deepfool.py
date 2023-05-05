import torch
import torchattacks
from tqdm import tqdm

def deepfool_attack(model, dataloader, device='cuda', batch_size=128):
    model = model.to(device)
    model.eval()
    atk = torchattacks.DeepFool(model, steps=3)

    adv_images = []
    true_labels = []
    pred_labels = []

    for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
        data, target = data.to(device), target.to(device)
        adv = atk(data, target)
        
        # Collect adversarial images and corresponding labels and predictions
        adv_images.append(adv.detach().cpu())
        true_labels.append(target.detach().cpu())
        pred_labels.append(torch.argmax(model(adv), dim=1).detach().cpu())

    # Concatenate all batches into a single tensor
    adv_images = torch.cat(adv_images, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)

    # Evaluate accuracy and print results
    accuracy = torch.sum(true_labels == pred_labels).item() / len(true_labels)
    print("Accuracy of the model on the adversarial examples: {:.2f}%".format(accuracy*100))

    return adv_images
