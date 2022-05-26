import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


class BaseAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.define_encoder()
        self.define_decoder()

    def define_encoder(self):
        raise NotImplementedError
    
    def define_decoder(self):
        raise NotImplementedError


class MNISTAutoEncoder(BaseAutoEncoder):
    def __init__(self):
        super().__init__()
    
    def define_encoder(self):
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))

    def define_decoder(self):
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


a = MNISTAutoEncoder()

# class LitAutoEncoder(pl.LightningModule):

#     def __init__(self):
#         super().__init__()
#         # self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
#         # self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

#         self.define_encoder()
#         self.define_decoder()

#     def define_encoder(self):
#         self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
    
#     def define_decoder(self):
#         self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

#     def forward(self, x):
#         # in lightning, forward defines the prediction/inference actions
#         embedding = self.encoder(x)
#         return embedding

#     def training_step(self, batch, batch_idx):
#         # training_step defined the train loop. It is independent of forward
#         x, y = batch
#         x = x.view(x.size(0), -1)
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         loss = F.mse_loss(x_hat, x)
#         self.log('train_loss', loss)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
    
# # dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
# # train, val = random_split(dataset, [55000, 5000])
from dataloader import load_mnist
train_dataloader, val_dataloader = load_mnist()

autoencoder = MNISTAutoEncoder()
#trainer = pl.Trainer()
trainer = pl.Trainer(max_epochs=1, gpus=1)
trainer.fit(autoencoder, train_dataloader, val_dataloader)    

# torchscript
autoencoder = MNISTAutoEncoder()
torch.jit.save(autoencoder.to_torchscript(), "model.pt")
print('training_finished')