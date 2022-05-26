"""
File to create a new FGSM attack on the latent space
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from dataloader import load_mnist
from models.autoencoder import ANNAutoencoder
from models.classifier import MNISTClassifier

# first load the autoencoder and the classifier
autoencoder_model = ANNAutoencoder.load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=9-step=9370.ckpt")
classifier_model  = MNISTClassifier.load_from_checkpoint("./lightning_logs/version_6/checkpoints/epoch=9-step=9370.ckpt")

train_dataloader, test_dataloader = load_mnist(batch_size=1)
criterion = nn.CrossEntropyLoss()

batch = next(iter(train_dataloader))
x, y = batch
x.requires_grad = True

z = autoencoder_model.get_z(x)
x_hat = autoencoder_model.get_x_hat(z)

# z.requires_grad = True
# x_hat.requires_grad = True

z.retain_grad()
x_hat.retain_grad()

preds = classifier_model(x_hat)
preds = F.log_softmax(preds, dim=-1)
print(preds.max(1, keepdim=True)[1][0][0])

loss = criterion(preds, y)
loss.backward()

# loss = -torch.gather(preds, 1, y.unsqueeze(dim=-1))
# loss.sum().backward()
# x_hat.grad = x.grad
# x_hat.backward(gradient=x.grad)

# print(x_hat.grad)
# print(z.grad)

noise_grad = torch.sign(z.grad)
z_perturbed = z + 0.2 * noise_grad
autoencoder_model.eval()
fake_img = autoencoder_model.get_x_hat(z_perturbed)

preds = classifier_model(fake_img)
preds = F.log_softmax(preds, dim=-1)
print(preds.max(1, keepdim=True)[1][0][0])

x_hat = x_hat.reshape(28, 28).cpu().detach().numpy()
fake_img = fake_img.reshape(28, 28).cpu().detach().numpy()
plt.gray()
fig, axis = plt.subplots(2)
axis[0].imshow(x_hat)
axis[1].imshow(fake_img)
plt.savefig(f"./img/fgsm_enc_dec2.png", dpi=600)
