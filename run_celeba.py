import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from dataloader import load_celeba
from models import CelebAAutoEncoder
from utils import save_checkpoint_autoencoder_new

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

##### constants #####
base_name = "celeba_autoencoder"
learning_rate = 1e-4
epochs = 10
#####################

print(f"Using {device} as the accelerator")

try:
    # try loading checkpoint
    checkpoint = torch.load(f'./models/{base_name}.pth.tar')
    print("Found Checkpoint :)")
    model = checkpoint["model"]
    model = model.to(device)

except:
    # train the model from scratch
    print("Couldn't find checkpoint :(")

    model = CelebAAutoEncoder(in_channels=3, dec_channels=32, latent_size=256)
    model = model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader, _ = load_celeba(batch_size=128)

    for epoch in range(epochs):
        for i, (image, _) in enumerate(train_dataloader):
            image = image.to(device)
            encoded_image, decoded_image = model(image)

            loss = criterion(decoded_image, image)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if i % 100 == 0 and i != 0:
                print(f"Epoch: [{epoch+1}][{i}/{len(train_dataloader)}] Loss: {loss.item(): .4f}")

        save_checkpoint_autoencoder_new(epoch, model, optimizer, path=f'./models/{base_name}.pth.tar')

# do reconstruction
_, test_dataloader = load_celeba(batch_size=1)
decoded_image      = None

for i, (image, _) in enumerate(test_dataloader):
    image = image.to(device)
    encoded_image, decoded_image = model(image)
    break

image = image.reshape(3, 128, 128).cpu().detach().numpy()
reconstructed_image = decoded_image.reshape(3, 128, 128).cpu().detach().numpy()

def denormalize(img, sigma, mu, n_channels=3):
    for i in range(n_channels):
        img[i] = (img[i] * sigma[i]) + mu[i]
    
    return img

# image = denormalize(image,
#                     sigma=(0.2023, 0.1994, 0.2010),
#                     mu=(0.4914, 0.4822, 0.4465))
# reconstructed_image = denormalize(reconstructed_image,
#                                     sigma=(0.2023, 0.1994, 0.2010),
#                                     mu=(0.4914, 0.4822, 0.4465))

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
