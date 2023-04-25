"""
Pytorch implementation of the Autoencoder and other models
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from utils import save_checkpoint_autoencoder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

##### constants #####
base_name = "cifar_autoencoder"

#####################

from torch import nn
from torchvision.utils import make_grid

from dataloader import load_cifar


def double_conv(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    # nn.Dropout(p=0.5), ## regualarisation..using high dropout rate of 0.9...lets see for few moments...
    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    # nn.Dropout(p=0.9) ## dual dropout 
  )

def down(in_channels, out_channels):
  ## downsampling with maxpool then double conv
  return nn.Sequential(
    nn.MaxPool2d(2),
    double_conv(in_channels, out_channels)
  )

class up(nn.Module):
  ## upsampling then double conv 
  def __init__(self, in_channels, out_channels):
    super(up, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride = 2)
    self.conv = double_conv(in_channels, out_channels)
  def forward(self,x1,x2): 
    x1 = self.up(x1)
    # input is CHW 
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2]) 
    return self.conv(x)

def inconv(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, kernel_size=1)

def outconv(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class Autoencoder(nn.Module):
    ...


class VAE(nn.Module):
  def __init__(self, n_channels=3, n_class=10, multi_channels=16, denoise=False):
    super().__init__()
    self.n_channels = n_channels
    self.n_class = n_class
    self.multi_channels = multi_channels 
    self.denoise = denoise
    
    
    if self.denoise: 
      layers = [] 
      layers.append(inconv(self.multi_channels*16*2, self.n_channels)) # *2 cause the input is gonna be now from the encoder of the reconstruction 
      layers.extend(
        [double_conv(self.n_channels, self.multi_channels),
        down(self.multi_channels,self.multi_channels*2),
        down(self.multi_channels*2,self.multi_channels*4),
        down(self.multi_channels*4,self.multi_channels*8), 
        down(self.multi_channels*8, self.multi_channels*16)]
      )
      self.encoder = nn.Sequential(*layers)
    
    else: 
      ## encoder defining   
      self.encoder = nn.Sequential(
        double_conv(self.n_channels, self.multi_channels),
        down(self.multi_channels,self.multi_channels*2),
        down(self.multi_channels*2,self.multi_channels*4),
        down(self.multi_channels*4,self.multi_channels*8), 
        down(self.multi_channels*8, self.multi_channels*16)
      )
        
    ## decoder defining
    self.decoder = nn.Sequential(
      up(self.multi_channels*16,self.multi_channels*8),
      up(self.multi_channels*8,self.multi_channels*4),
      up(self.multi_channels*4,self.multi_channels*2),
      up(self.multi_channels*2,self.multi_channels),
      outconv(self.multi_channels,self.n_class)
    )
  
  def forward(self, x):  
    if self.denoise:
      x = self.encoder[0](x) 
      x1 = self.encoder[1](x)
      x2 = self.encoder[2](x1)
      x3 = self.encoder[3](x2)
      x4 = self.encoder[4](x3)
      x5 = self.encoder[5](x4)
    else: 
      x1 = self.encoder[0](x)
      x2 = self.encoder[1](x1)
      x3 = self.encoder[2](x2)
      x4 = self.encoder[3](x3)
      x5 = self.encoder[4](x4) 
    
    # pdb.set_trace()
    x = self.decoder[0](x5,x4)
    x = self.decoder[1](x,x3)
    x = self.decoder[2](x,x2)
    x = self.decoder[3](x,x1)
    logits = self.decoder[4](x)  
    return logits


class CifarEncoder(nn.Module):
    
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


class CifarDecoder(nn.Module):
    
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


class CelebAAutoEncoder(nn.Module):
    def __init__(self, in_channels, dec_channels, latent_size):
        super(CelebAAutoEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.latent_size = latent_size

        ###############
        # ENCODER
        ##############
        self.e_conv_1 = nn.Conv2d(in_channels, dec_channels, 
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_1 = nn.BatchNorm2d(dec_channels)

        self.e_conv_2 = nn.Conv2d(dec_channels, dec_channels*2, 
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_2 = nn.BatchNorm2d(dec_channels*2)

        self.e_conv_3 = nn.Conv2d(dec_channels*2, dec_channels*4, 
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_3 = nn.BatchNorm2d(dec_channels*4)

        self.e_conv_4 = nn.Conv2d(dec_channels*4, dec_channels*8, 
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_4 = nn.BatchNorm2d(dec_channels*8)

        self.e_conv_5 = nn.Conv2d(dec_channels*8, dec_channels*16, 
                                  kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.e_bn_5 = nn.BatchNorm2d(dec_channels*16)
       
        self.e_fc_1 = nn.Linear(dec_channels*16*4*4, latent_size)

        ###############
        # DECODER
        ##############
        
        self.d_fc_1 = nn.Linear(latent_size, dec_channels*16*4*4)

        self.d_conv_1 = nn.Conv2d(dec_channels*16, dec_channels*8, 
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_1 = nn.BatchNorm2d(dec_channels*8)

        self.d_conv_2 = nn.Conv2d(dec_channels*8, dec_channels*4, 
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_2 = nn.BatchNorm2d(dec_channels*4)

        self.d_conv_3 = nn.Conv2d(dec_channels*4, dec_channels*2, 
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_3 = nn.BatchNorm2d(dec_channels*2)

        self.d_conv_4 = nn.Conv2d(dec_channels*2, dec_channels, 
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_4 = nn.BatchNorm2d(dec_channels)
        
        self.d_conv_5 = nn.Conv2d(dec_channels, in_channels, 
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        
        
        # Reinitialize weights using He initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()


    def encode(self, x):
        
        #h1
        x = self.e_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.e_bn_1(x)
        
        #h2
        x = self.e_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)    
        x = self.e_bn_2(x)     

        #h3
        x = self.e_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
        x = self.e_bn_3(x)
        
        #h4
        x = self.e_conv_4(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
        x = self.e_bn_4(x)
        
        #h5
        x = self.e_conv_5(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True) 
        x = self.e_bn_5(x)        
        
        #fc
        x = x.view(-1, self.dec_channels*16*4*4)
        x = self.e_fc_1(x)
        return x

    def decode(self, x):
        
        # h1
        #x = x.view(-1, self.latent_size, 1, 1)
        x = self.d_fc_1(x)
        
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)  
        x = x.view(-1, self.dec_channels*16, 4, 4) 

        
        # h2
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_1(x)
        
        # h3
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_2(x)
        
        # h4
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_3(x)  

        # h5
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_4(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_4(x)
        
        
        # out
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_5(x)
        x = torch.sigmoid(x)
        
        return x

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)
        return z, decoded


class MNISTVAE(nn.Module):
    ...


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

        encoder = CifarEncoder(num_input_channels=3, base_channel_size=32, latent_dim=256)
        decoder = CifarDecoder(num_input_channels=3, base_channel_size=32, latent_dim=256)
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

            save_checkpoint_autoencoder(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, path=f'./models/{base_name}.pth.tar')
    
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
