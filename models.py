"""
Pytorch implementation of the Autoencoder
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
base_name = "cifar_autoencoder"

from torch import nn
from torchvision.utils import make_grid
from dataloader import load_cifar
	
def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, path=f'./models/{base_name}.pth.tar'):
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}

    filename = path
    torch.save(state, filename)


class Encoder(nn.Module):
    
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 act_fn : object = nn.GELU):
        """
        Inputs: 
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 act_fn : object = nn.GELU):
        """
        Inputs: 
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x

if __name__ == "__main__":
    print(f"Using {device} as the accelerator")
    epochs = 10

    try:
        # try loading checkpoint
        checkpoint = torch.load(f'./models/{base_name}.pth.tar')
        print("Found Checkpoint :)")
        encoder = checkpoint["encoder"]
        decoder = checkpoint["decoder"]
        encoder.to(device)
        decoder.to(device)

    except:
        # train the model from scratch
        print("Couldn't find checkpoint :(")

        encoder = Encoder(num_input_channels=3, base_channel_size=32, latent_dim=256)
        decoder = Decoder(num_input_channels=3, base_channel_size=32, latent_dim=256)
        encoder.to(device)
        decoder.to(device)
        criterion = nn.MSELoss().to(device)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-5)
        
        train_dataloader, _, _ = load_cifar()

        for epoch in range(epochs):
            for i, (image, _) in enumerate(train_dataloader):
                image.to(device)
                encoded_image = encoder(image)
                decoded_image = decoder(encoded_image)

                # print(encoded_image)
                # print(decoded_image)
                
                loss = criterion(decoded_image, image)
                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss.backward()

                decoder_optimizer.step()
                encoder_optimizer.step()

                if i % 100 == 0 and i != 0:
                    print(f"Epoch: [{epoch+1}][{i}/{len(train_dataloader)}] Loss: {loss.item(): .4f}")

            save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer)
    
    # do reconstruction
    _, _, test_dataloader = load_cifar(batch_size=1)
    decoded_image    = None

    for i, (image, _) in enumerate(test_dataloader):
        image.to(device)
        encoded_image = encoder(image)
        decoded_image = decoder(encoded_image)
        break
    
    image = image.reshape(3, 32, 32).detach().numpy()
    reconstructed_image = decoded_image.reshape(3, 32, 32).detach().numpy()

    def denormalize(img, sigma, mu, n_channels=3):
        for i in range(n_channels):
            img[i] = (img[i] * sigma[i]) + mu[i]
        
        return img

    image = denormalize(image,
                        sigma=(0.2023, 0.1994, 0.2010),
                        mu=(0.4914, 0.4822, 0.4465))
    reconstructed_image = denormalize(reconstructed_image,
                                      sigma=(0.2023, 0.1994, 0.2010),
                                      mu=(0.4914, 0.4822, 0.4465))
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    fig, axis = plt.subplots(1,2)
    print(image.shape)
    axis[0].imshow(np.transpose(image, (1, 2, 0)))
    axis[1].imshow(np.transpose(reconstructed_image, (1, 2, 0)))
    plt.savefig(f"./img/{base_name}_recon.png", dpi=1000)
    plt.show()
