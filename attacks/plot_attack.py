from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def plot_robust_accuracy(result: Dict, plot_dir: str, accuracy: float, filename: str) -> bool:
    is_success = False
    labels, original, modified, hybrid = ["benign"], [accuracy * 100], [0], [0]
    for attack_name, robust_acc in result.items():
        labels.append(attack_name)
        original.append(robust_acc["original"] * 100)
        modified.append(robust_acc["modified"] * 100)
        hybrid.append(robust_acc["hybrid"]*100)
    
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    ax.bar(x[0], original[0], width)
    rects1 = ax.bar(x[1:] - width, original[1:], width, label='Original')
    rects2 = ax.bar(x[1:], modified[1:], width, label='Modified')
    rects3 = ax.bar(x[1:] + width, hybrid[1:], width, label="Hybrid")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(0, 100)
    ax.set_ylabel('Robust Accuracy')
    # ax.set_title('Method v/s Robust Accuracy Plot')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.savefig(f"{plot_dir}/{filename}")
    is_success = True

    return 

def plot_adversarial_images(result: Dict, plot_dir: str, filename: str) -> bool:
    """
    For a given dataset, take all the class and plot original, recon,
    original adversarial attack and modified adversarial attack, and
    their respective noise levels.
    """
    orig_attk_images = result["FastGradientMethod"]["x_test_adv_np"]
    modf_attk_images = result["FastGradientMethod"]["hybrid_x_np"]

    plt.figure(figsize=(5, 5))
    images = torch.Tensor(modf_attk_images).reshape(-1, 28, 28)
    i = 1
    for image in images:
        plt.subplot(5, 5, i, xticks=[], yticks=[])
        plt.imshow(image)
        i = i + 1

        if i == 26:
            # print(image)
            break
    
    plt.savefig(f"{plot_dir}/{filename}")
    # grid_images = torchvision.utils.make_grid(images, nrow=10)
    # print(grid_images.shape)

    # plt.imshow(grid_images.permute(1, 2, 0))