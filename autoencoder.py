"""
Pytorch implementation of the Autoencoder
"""
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from torch import nn
from dataloader import load_mnist
	
def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, path='./models/checkpoint_enc_dec.pth.tar'):
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}

    filename = path
    torch.save(state, filename)


class Encoder(nn.Module):
    """
    Encoder to encode the image into a hidden state
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.Mish(),
            nn.Linear(512, 264),
            nn.Mish(),
            nn.Linear(264, 128),
            nn.Mish()
        )
    
    def forward(self, images):
        out = self.model(images)
        
        return out


class Decoder(nn.Module):
    """
    Decoder to reconstruct image from the hidden state
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 264),
            nn.Mish(),
            nn.Linear(264, 512),
            nn.Mish(),
            nn.Linear(512, 784),
            nn.Sigmoid() # for making value between 0 to 1
        )
    
    def forward(self, encoded_image):
        out = self.model(encoded_image)

        return out

if __name__ == "__main__":
    print(f"Using {device} as the accelerator")
    try:
        # try loading checkpoint
        checkpoint = torch.load('./models/checkpoint_enc_dec.pth.tar')
        print("Found Checkpoint :)")
        encoder = checkpoint["encoder"]
        decoder = checkpoint["decoder"]
        encoder.to(device)
        decoder.to(device)

    except:
        # train the model from scratch
        print("Couldn't find checkpoint :(")

        encoder = Encoder()
        decoder = Decoder()
        encoder.to(device)
        decoder.to(device)
        criterion = nn.MSELoss().to(device)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-5)
        
        train_dataloader, _ = load_mnist()

        for epoch in range(10):
            for i, (image, _) in enumerate(train_dataloader):
                image.to(device)
                encoded_image = encoder(image)
                decoded_image = decoder(encoded_image)
                
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
    _, test_dataloader = load_mnist(batch_size=1)
    decoded_image    = None

    for i, (image, _) in enumerate(test_dataloader):
        image.to(device)
        encoded_image = encoder(image)
        decoded_image = decoder(encoded_image)
        break
    
    image = image.reshape(28, 28).detach().numpy()
    reconstructed_image = decoded_image.reshape(28, 28).detach().numpy()

    plt.gray()
    fig, axis = plt.subplots(2)
    axis[0].imshow(image)
    axis[1].imshow(reconstructed_image)
    plt.savefig(f"./img/enc_dec.png", dpi=600)