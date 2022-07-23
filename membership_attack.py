import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from dataloader import load_mnist
from autoencoder import EncoderMember, DecoderMember

def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, path='./checkpoints/membership_enc_dec1.pth.tar'):
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}

    filename = path
    torch.save(state, filename)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if __name__ == "__main__":
    print(f"Using {device} as the accelerator")

    #####
    epochs = 10
    #####

    try:
        # try loading checkpoint
        checkpoint = torch.load('./checkpoints/membership_enc_dec1.pth.tar')
        print("Found Checkpoint :)")
        encoder = checkpoint["encoder"]
        decoder = checkpoint["decoder"]
        encoder.to(device)
        decoder.to(device)

    except:
        # train the model from scratch
        print("Couldn't find checkpoint :(")
        encoder = EncoderMember(num_classes=10).to(device)
        decoder = DecoderMember(num_classes=10).to(device)

        recon_criterion = nn.MSELoss().to(device)
        cls_criterion   = nn.CrossEntropyLoss().to(device)

        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-5)
        
        train_dataloader, _, _ = load_mnist()

        for epoch in range(epochs):
            for i, (image, label) in enumerate(train_dataloader):
                image = image.to(device)
                label = label.to(device)
                encoded_image = encoder(image)
                decoded_image = decoder(encoded_image)

                recon_loss = recon_criterion(image, decoded_image)
                cls_loss   = cls_criterion(encoded_image, label)

                loss = recon_loss + cls_loss

                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()

                loss.backward(retain_graph=True)

                decoder_optimizer.step()
                encoder_optimizer.step()

                # encoder_optimizer.zero_grad()
                # cls_loss.backward()
                # encoder_optimizer.step()

                if i % 100 == 0 and i != 0:
                    print(f"Epoch: [{epoch+1}][{i}/{len(train_dataloader)}] Reconstruction Loss: {recon_loss.item(): .4f} Classification Loss: {cls_loss.item(): .4f}")
        
        save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer)

    _, _, test_dataloader = load_mnist(batch_size=1)
    for i, (image, label) in enumerate(test_dataloader):
        image = image.to(device)
        label = label.to(device)

        z = encoder(image)
        print(f"Actual z: {z}")
        recon_image = decoder(z)

        break
    
    print(f"Label is: {label}")
    # import pdb; pdb.set_trace()
    z = z.detach().reshape(10)

    # demo_z = torch.rand(10).to(device)
    # demo_z[label] += 0.9
    # demo_z = torch.tensor([-5.4, -3.2, -4.1, -7.1, 0.1, 0, -4.53, 2, -2.3, 0.1]).to(device)
    # demo_z[label] += 20
    # demo_z[label+1] += 20
    # demo_z = nn.Softmax(demo_z)
    # print(f"Crafted z: {demo_z}")

    image = image.reshape(28, 28).cpu().detach().numpy()
    reconstructed_image = recon_image.reshape(28, 28).cpu().detach().numpy()
    # plt.subplot(10, 10, 1, xticks=[], yticks=[])
    # plt.imshow(reconstructed_image)

    plt.gray()
    plt.figure(figsize=(16, 16))
    for i in range(10):
        for j in range(1, 11):
            plt.subplot(10, 10, i*10+j, xticks=[], yticks=[])
            z[(label+i) % 10] += 2
            # print(f"Crafted z: {z}")
            demo_recon = decoder(z)
            demo_image = demo_recon.reshape(28, 28).cpu().detach().numpy()
            plt.imshow(demo_image)
        z[(label+i) % 10] -= 20

    # fig, axis = plt.subplots(2)
    # axis[0].imshow(reconstructed_image)
    # axis[1].imshow(demo_image)
    plt.savefig(f"./img/membership_enc_dec.png", dpi=600)
