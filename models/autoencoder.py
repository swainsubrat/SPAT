"""
Pytorch-Lightning implementation of the vanilla autoencoder
"""
import sys
sys.path.append("..")

import torch
import torchvision

import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from utils import visualize_cifar_reconstructions
from dataloader import load_mnist, load_cifar, load_fashion_mnist


class BaseAutoEncoder(pl.LightningModule):
    """
    Base class for all the autoencoders
    """
    def __init__(self):
        super().__init__()
        self.define_encoder()
        self.define_decoder()

    def define_encoder(self):
        """
        Define encoder by overriding this function
        """
        raise NotImplementedError

    def define_decoder(self):
        """
        Define decoder by overriding this function
        """
        raise NotImplementedError

    def get_z(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding of the input

        Args:
            x (torch.Tensor): Input image to the encoder
        
        Returns:
            Encoded input
        """
        return self.encoder(x)

    def get_x_hat(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get the reconstructed output

        Args:
            z (torch.Tensor): Image embeddings
        
        Returns:
            Reconstructed output
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, z

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class ANNAutoencoder(BaseAutoEncoder):
    """
    This is an implementation of the autoencoder using ANN
    """
    def __init__(self,
                 input_dim: int=784,
                 latent_dim: int=128,
                 activation_fn: nn.modules.activation=nn.Mish) -> None:
        """
        Args:
            input_dim (int): Dimension of the input to the autoencoder
            latent_dim (int): Dimension of the latent dimension
            activation_fn (nn.modules.activation): Activation function 
        """
        self.save_hyperparameters()
        self.input_dim     = input_dim
        self.latent_dim    = latent_dim
        self.activation_fn = activation_fn
        super().__init__()

    def define_encoder(self) -> None:
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            self.activation_fn(),
            nn.Linear(512, 264),
            self.activation_fn(),
            nn.Linear(264, self.latent_dim),
            self.activation_fn()
        )

    def define_decoder(self) -> None:
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 264),
            self.activation_fn(),
            nn.Linear(264, 512),
            self.activation_fn(),
            nn.Linear(512, 784),
            nn.Sigmoid() # for making value between 0 to 1
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat, _ = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True,\
                 prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat, _ = self(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        self.log("valid_loss", loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch

        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}


class CIFAR10Autoencoder(BaseAutoEncoder):
    """
    This is an implementation of the autoencoder for CIFAR10
    """
    def __init__(self,
                 input_dim: int=784,
                 latent_dim: int=128,
                 num_input_channels: int=3,
                 base_channel_size: int=32,
                 activation_fn: nn.modules.activation=nn.GELU) -> None:
        """
        Args:
            input_dim (int): Dimension of the input to the autoencoder
            latent_dim (int): Dimension of the latent dimension
            num_input_channels (int): Number of input channels of the image.
            For CIFAR, this parameter is 3
            base_channel_size : Number of channels we use in the first 
            convolutional layers. Deeper layers might use a duplicate of it.
            activation_fn (nn.modules.activation): Activation function 
        """
        self.save_hyperparameters()
        self.input_dim     = input_dim
        self.latent_dim    = latent_dim
        self.num_input_channels = num_input_channels
        self.c_hid         = base_channel_size
        self.activation_fn = activation_fn
        super().__init__()
        # self.linear        = nn.Sequential(
        #     nn.Linear(self.latent_dim, 2 * 16 * self.c_hid),
        #     self.activation_fn()
        # )
    
    def define_encoder(self) -> None:
        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            self.activation_fn(),
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, padding=1),
            self.activation_fn(),
            nn.Conv2d(self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            self.activation_fn(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1),
            self.activation_fn(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            self.activation_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * self.c_hid, self.latent_dim),
        )

    def define_decoder(self) -> None:
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2 * 16 * self.c_hid),
            nn.Unflatten(1, (-1, 4, 4)),
            self.activation_fn(),
            nn.ConvTranspose2d(
                2 * self.c_hid, 2 * self.c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            self.activation_fn(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1),
            self.activation_fn(),
            nn.ConvTranspose2d(2 * self.c_hid, self.c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            self.activation_fn(),
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, padding=1),
            self.activation_fn(),
            nn.ConvTranspose2d(
                self.c_hid, self.num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        # z = self.linear(z)
        # z = z.reshape(z.shape[0], -1, 4, 4)
        x_hat = self.decoder(z)

        return x_hat, z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, _ = self(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        # if batch_idx == 1: print(loss);
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, _ = self(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        # if batch_idx == 1: print(loss);
        self.log("valid_loss", loss)
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}


if __name__ == "__main__":

    """
    Testing CIFAR autoencoder
    """
    # import os
    # import torchsummary

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 4"
    # train_dataloader, valid_dataloader, test_dataloader = load_cifar(
    #     root="/home/sweta/scratch/datasets/CIFAR/", batch_size=128
    # )
    
    # model = CIFAR10Autoencoder()
    # torchsummary.summary(model, (3, 32, 32))
    # # trainer = pl.Trainer(max_epochs=500, gpus=2, default_root_dir="..")
    # trainer = pl.Trainer(max_epochs=200, gpus=2, default_root_dir="..")
    # trainer.fit(model, train_dataloader, valid_dataloader)

    # Testing
    # train_dataloader, valid_dataloader, test_dataloader = load_cifar(
    #     root="/home/sweta/scratch/datasets/CIFAR/", batch_size=4
    # )
    # input_imgs, _ = next(iter(test_dataloader))
    # model = CIFAR10Autoencoder.load_from_checkpoint("../lightning_logs/version_9/checkpoints/epoch=199-step=35000.ckpt")
    # model.eval()
    # reconst_imgs, _ = model(input_imgs)

    # input_imgs   = input_imgs.reshape(3, 32, 32).detach()
    # reconst_imgs = reconst_imgs.reshape(3, 32, 32).detach()
    # fig, axis = plt.subplots(1,2)

    # axis[0].imshow(np.transpose(input_imgs, (1, 2, 0)))
    # axis[1].imshow(np.transpose(reconst_imgs, (1, 2, 0)))
    
    # plt.savefig(f"../img/recons/new_recon.png", dpi=1000)
    # plt.show()
    # visualize_cifar_reconstructions(input_imgs, reconst_imgs)

    """
    Testing MNIST autoencoder
    """
    train_dataloader, valid_dataloader, test_dataloader = load_mnist(
        root="/home/sweta/scratch/datasets/MNIST/", batch_size=128
    )
    model = ANNAutoencoder()
    trainer = pl.Trainer(max_epochs=10, gpus=1, default_root_dir="..")
    # trainer.fit(model, train_dataloader)    

    model = ANNAutoencoder.load_from_checkpoint("../lightning_logs/version_0/checkpoints/epoch=9-step=9370.ckpt")
    model.eval()
    preds = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True)
    print(len(preds))

    """
    Testing FashionMNIST autoencoder
    """
    # train_dataloader, valid_dataloader, test_dataloader = load_fashion_mnist(
    #     root="/home/sweta/scratch/datasets/FashionMNIST/", batch_size=128
    # )
    # model = ANNAutoencoder()
    # trainer = pl.Trainer(max_epochs=20, gpus=1, default_root_dir="..")
    # # trainer.fit(model, train_dataloader, valid_dataloader)    

    # train_dataloader, valid_dataloader, test_dataloader = load_fashion_mnist(
    #     root="/home/sweta/scratch/datasets/FashionMNIST/", batch_size=1
    # )
    # model = ANNAutoencoder.load_from_checkpoint("../lightning_logs/version_12/checkpoints/epoch=19-step=8580.ckpt")
    # model.eval()
    
    # images, labels = next(iter(train_dataloader))
    # recons, _      = model(images)
    # images = images.reshape(28, 28).cpu().detach().numpy()
    # recons = recons.reshape(28, 28).cpu().detach().numpy()

    # plt.gray()
    # fig, axis = plt.subplots(2)
    # axis[0].imshow(images)
    # axis[1].imshow(recons)
    # plt.savefig(f"../img/recons/fmnist_enc_dec.png", dpi=600)
