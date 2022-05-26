import foolbox as fb

from models.classifier import MNISTClassifier
from foolbox.utils import accuracy, samples

# first load the autoencoder and the classifier
model  = MNISTClassifier.load_from_checkpoint("./lightning_logs/version_6/checkpoints/epoch=9-step=9370.ckpt")
model.eval()

# model = ...

print(type(model))
fmodel = fb.PyTorchModel(model, bounds=(0, 1))

attack = fb.attacks.LinfPGD()
epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
images, labels = samples(fmodel, dataset="mnist", batchsize=20)

print(fmodel)

print(type(images), type(labels))
print(images.requires_grad, labels.shape)

images = images.reshape((20, -1))

_, advs, success = attack(fmodel, images, labels, epsilons=epsilons)

print(success)
