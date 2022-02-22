"""
Pytorch implementation of the Autoencoder
"""
import torch
import matplotlib.pyplot as plt

from ga import generate_perturbation
from autoencoder import Encoder, Decoder, save_checkpoint

from torch import nn
from dataloader import load_mnist

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Using {device} as the accelerator")

encoder = Encoder()
decoder = Decoder()
encoder.to(device)
decoder.to(device)
criterion = nn.MSELoss().to(device)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-5)

train_dataloader, _ = load_mnist()

for epoch in range(1):
    for i, (image, _) in enumerate(train_dataloader):
        image.to(device)
        encoded_image = encoder(image)

        # Add perturbation to the latent space
        delta = generate_perturbation(encoded_image.shape)
        adv_encoded_image = encoded_image + delta

        decoded_image = decoder(adv_encoded_image)
        
        # Reconstruction loss + Perturbation in the Adversarial Space
        loss_recon = criterion(decoded_image, image)
        loss_adv   = criterion(encoded_image, adv_encoded_image)
        loss = loss_recon + loss_adv

        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()

        decoder_optimizer.step()
        encoder_optimizer.step()

        break

        # if i % 100 == 0 and i != 0:
        #     print(f"Epoch: [{epoch+1}][{i}/{len(train_dataloader)}] Loss: {loss.item(): .4f}")

    # save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer)

# try:
#     # try loading checkpoint
#     checkpoint = torch.load('./models/checkpoint_enc_dec.pth.tar')
#     print("Found Checkpoint :)")
#     encoder = checkpoint["encoder"]
#     decoder = checkpoint["decoder"]
#     encoder.to(device)
#     decoder.to(device)

# except:
#     # train the model from scratch
#     print("Couldn't find checkpoint :(")

#     encoder = Encoder()
#     decoder = Decoder()
#     encoder.to(device)
#     decoder.to(device)
#     criterion = nn.MSELoss().to(device)
#     encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
#     decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-5)
    
#     train_dataloader, _ = load_mnist()

#     for epoch in range(10):
#         for i, (image, _) in enumerate(train_dataloader):
#             image.to(device)
#             encoded_image = encoder(image)
#             decoded_image = decoder(encoded_image)
            
#             loss = criterion(decoded_image, image)
#             decoder_optimizer.zero_grad()
#             encoder_optimizer.zero_grad()
#             loss.backward()

#             decoder_optimizer.step()
#             encoder_optimizer.step()

#             if i % 100 == 0 and i != 0:
#                 print(f"Epoch: [{epoch+1}][{i}/{len(train_dataloader)}] Loss: {loss.item(): .4f}")

#         save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer)

# # do reconstruction
# _, test_dataloader = load_mnist(batch_size=1)
# decoded_image    = None

# for i, (image, _) in enumerate(test_dataloader):
#     image.to(device)
#     encoded_image = encoder(image)
#     decoded_image = decoder(encoded_image)
#     break

# image = image.reshape(28, 28).detach().numpy()
# reconstructed_image = decoded_image.reshape(28, 28).detach().numpy()

# plt.gray()
# fig, axis = plt.subplots(2)
# axis[0].imshow(image)
# axis[1].imshow(reconstructed_image)
# plt.show()
# plt.savefig(f"./img/enc_dec.png", dpi=600)