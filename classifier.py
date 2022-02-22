"""
Pytorch implementation of the Autoencoder
"""
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from torch import nn
	
def save_checkpoint(epoch, classifier, optimizer, path='./models/checkpoint_mlp.pth.tar'):
    state = {'epoch': epoch,
             'classifier': classifier,
             'optimizer': optimizer}

    filename = path
    torch.save(state, filename)


class MLP(nn.Module):
    """
    Encoder to encode the image into a hidden state
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    
    def forward(self, images):
        out = self.model(images)
        
        return out
