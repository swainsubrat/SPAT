import torch
import torchvision
import numpy as np
import pickle
import matplotlib.pyplot as plt

from typing import List, Dict

def accuracy(Y: List, predY: List) -> float:
    """
    Get accuracy
    """
    Y = np.array(Y)
    predY = np.array(predY)
    accuracy = (Y == predY).sum()/ float(len(Y))
    accuracy = np.round(accuracy * 100, 2)

    return accuracy

def save(path: str, params: Dict) -> None:
    """
    Save model to path
    """
    outfile = open(path, 'wb')
    pickle.dump(params, outfile)
    outfile.close()

def load(path: str) -> Dict:
    """
    Load model from path
    """
    infile = open(path, 'rb')
    params = pickle.load(infile)
    infile.close()

    return params

def save_checkpoint_autoencoder(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, path):
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}

    filename = path
    torch.save(state, filename)

def save_checkpoint_autoencoder_new(epoch, model, optimizer, path):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}

    filename = path
    torch.save(state, filename)

def visualize_cifar_reconstructions(input_imgs, reconst_imgs):
    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7, 4.5))
    plt.title("Reconstructed image from the latent codes")
    plt.imshow(grid)
    plt.axis("off")
    # plt.show()
    plt.savefig(f"../img/recons/cifar10_3.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    a = [1, 2, 3]
    b = [1, 2, 3]
    print(accuracy(a, b))