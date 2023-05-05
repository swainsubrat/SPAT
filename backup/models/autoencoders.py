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


class ANNVAEGAN(BaseAutoEncoder):
    """
    This is an implementation of the autoencoder using ANN
    """
    def __init__(self,
                 input_dim: int=784,
                 latent_dim: int=128,
                 activation_fn: nn.modules.activation=nn.ReLU) -> None:
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
        self.fc_mean       = nn.Linear(256, self.latent_dim)
        self.fc_logvar     = nn.Linear(256, self.latent_dim)
        super().__init__()

    def define_encoder(self) -> None:
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            self.activation_fn(),
            nn.Linear(512, 264),
            self.activation_fn(),
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
    
    def define_discriminator(self) -> None:
        self.discriminator = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # encoder part
        out = self.encoder(x)
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        std = logvar.mul(0.5).exp_()

        # sampling and decoding
        epsilon = torch.rand_like()
        z = mean + std * epsilon
        x_hat = self.decoder(z)

        return x_hat, z

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat, _ = self(x)
        loss = F.mse_loss(x_hat, x, reduction="none").sum(dim=[1]).mean(dim=[0])
        self.log("train_loss", loss, on_step=True, on_epoch=True,\
                prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat, _ = self(x)
        loss = F.mse_loss(x, x_hat, reduction="none").sum(dim=[1]).mean(dim=[0])
        self.log("valid_loss", loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch

        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}


class FashionMNISTAutoencoder(BaseAutoEncoder):
    """
    This is an implementation of the autoencoder for Fashion MNIST
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
        x = x.view(x.size(0), -1)
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


class CelebAAutoencoder(BaseAutoEncoder):
    """
    This is an implementation of the autoencoder for CelebA
    """
    def __init__(self,
                # input_dim: int=784,
                # latent_dim: int=128,
                lr: float=1e-4,
                num_input_channels: int=3,
                num_output_channels: int=3,
                base_channel_size: int=16,
                activation_fn: nn.modules.activation=nn.ReLU) -> None:
        """
        Args:
            # input_dim (int): Dimension of the input to the autoencoder
            # latent_dim (int): Dimension of the latent dimension
            num_input_channels (int): Number of input channels of the image.
            For CIFAR, this parameter is 3
            num_output_channels (int): Number of output channels of the image.
            For CIFAR, this parameter is 3
            base_channel_size : Number of channels we use in the first 
            convolutional layers. Deeper layers might use a duplicate of it.
            activation_fn (nn.modules.activation): Activation function 
        """
        self.save_hyperparameters()
        self.lr = lr
        self.in_chs   = num_input_channels
        self.out_chs  = num_output_channels
        self.base_chs = base_channel_size
        self.activation_fn = activation_fn
        super().__init__()

    def define_encoder(self):
        self.encoder = nn.Sequential(
            double_conv(self.in_chs, self.base_chs),
            down(self.base_chs,self.base_chs*2),
            down(self.base_chs*2,self.base_chs*4),
            down(self.base_chs*4,self.base_chs*8), 
        )

    def define_decoder(self):
        self.decoder = nn.Sequential(
            up(self.base_chs*8,self.base_chs*4),
            up(self.base_chs*4,self.base_chs*2),
            up(self.base_chs*2,self.base_chs),
            outconv(self.base_chs,self.out_chs)
        )
    
    def get_z(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)

        return x4, x3, x2, x1
    
    def get_x_hat(self, x4, x3, x2, x1) -> torch.Tensor:
        x = self.decoder[0](x4,x3)
        x = self.decoder[1](x,x2)
        x = self.decoder[2](x,x1)
        logits = self.decoder[3](x)
        outputs = F.sigmoid(logits)

        return outputs

    def forward(self, x):
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3) 
        
        x = self.decoder[0](x4,x3)
        x = self.decoder[1](x,x2)
        x = self.decoder[2](x,x1)
        logits = self.decoder[3](x)
        outputs = F.sigmoid(logits)

        return outputs

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss_bce = F.binary_cross_entropy_with_logits(x, x_hat, reduction="none")
        loss_mse = F.mse_loss(x, x_hat, reduction="none")
        loss = (0.2 * loss_bce) + (0.8 * loss_mse)
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        self.log("train_loss", loss)

        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     x, _ = batch
    #     x_hat, _ = self(x)
    #     loss = F.mse_loss(x, x_hat, reduction="none")
    #     loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    #     # if batch_idx == 1: print(loss);
    #     self.log("valid_loss", loss)
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        
        return self(x)

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}


if __name__ == "__main__":
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
    # model = ANNAutoencoder.load_from_checkpoint("../lightning_logs/fmnist_ae_mse/checkpoints/epoch=19-step=8580.ckpt")
    # model.eval()
    
    # images, labels = next(iter(train_dataloader))
    # recons, _      = model(images)
    # images = images.reshape(28, 28).cpu().detach().numpy()
    # recons = recons.reshape(28, 28).cpu().detach().numpy()

    # plt.gray()
    # fig, axis = plt.subplots(2)
    # axis[0].imshow(images)
    # axis[1].imshow(recons)
    # plt.savefig(f"../plots/fmnist_enc_dec.png", dpi=600)

    """
    Testing class-constrained MNIST autoencoder
    """
    # train_dataloader, valid_dataloader, test_dataloader = load_mnist(
    #     root="~/scratch/datasets/MNIST/", batch_size=128
    # )
    # model = ClassConstrainedANNAutoencoder()
    # trainer = pl.Trainer(max_epochs=10, accelerator="mps", default_root_dir="..")
    # # trainer.fit(model, train_dataloader, valid_dataloader)    

    # model = ClassConstrainedANNAutoencoder.load_from_checkpoint("../lightning_logs/version_39/checkpoints/checkpoint.ckpt")
    # model.eval()

    # # Reconstruct
    # train_dataloader, valid_dataloader, test_dataloader = load_mnist(
    #     root="~/scratch/datasets/MNIST/", batch_size=10
    # )
    # x, y = next(iter(test_dataloader))
    # x = x.to(device)
    # model = model.to(device)
    # x_hat, z, preds = model(x)
    # # make_dot(preds, params=dict(list(model.named_parameters()))).render("torchviz", format="png")
    # # p = trainer.test(model, dataloaders=test_dataloader, verbose=True)
    # # print(p)
    # x = x.cpu().detach().numpy()
    # x_hat = x_hat.cpu().detach().numpy()
    # # images = np.concatenate([x, x_hat], 0)
    # # plot_recons(images=images, reshape=(-1, 1, 28, 28))
    # for image, code, recon in zip(x, z, x_hat):
    #     plot_recons_with_latent_codes(image, recon, code)

    # # train_dataloader, valid_dataloader, test_dataloader = load_mnist(batch_size=128)
    # # model = ClassConstrainedANNAutoencoder()
    # # trainer = pl.Trainer(max_epochs=10, gpus=1, default_root_dir="..", checkpoint_callback=True, logger=True)
    # # # trainer.fit(model, train_dataloader, valid_dataloader)    

    # # model = ClassConstrainedANNAutoencoder.load_from_checkpoint("../lightning_logs/version_14/checkpoints/epoch=9-step=4290.ckpt")
    # # model = model.to(device)
    # # model.eval()

    # _, _, test_dataloader = load_mnist(batch_size=1)
    # encoded_samples = []
    # for i, (image, label) in enumerate(test_dataloader):
    #     image = image.to(device)
    #     label = label.item()

    #     with torch.no_grad():
    #         z = model.get_z(image)
    #     encoded_img = z.flatten().cpu().numpy()
    #     encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
    #     encoded_sample['label'] = label
    #     encoded_samples.append(encoded_sample)

    # encoded_samples = pd.DataFrame(encoded_samples)
    # # print(encoded_samples)
    # tsne = TSNE(n_components=2)
    # tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))

    # fig = px.scatter(tsne_results, x=0, y=1,
    #                 color=encoded_samples.label.astype(str),
    #                 color_discrete_map={"0":"red", "1":"blue", "2":"yellow", "3":"gray", "4":"brown", "5":"aqua", "6":"maroon", "7":"purple", "8":"teal", "9":"lime"},
    #                 labels={'0': 'dimension-1', '1': 'dimension-2'})
    # # fig.write_image("../img/tsne_cc_extended.png")
    # fig.show()
