from models.autoencoder import (ANNAutoencoder, ClassConstrainedANNAutoencoder,
                                CelebAAutoencoder, CIFAR10Autoencoder)
from models.classifier import (MNISTClassifier, CelebAClassifier, CIFAR10Classifier)

MODEL_MAPPINGS = {
    "./lightning_logs/mnist_classifier/checkpoints/epoch=9-step=9370.ckpt": MNISTClassifier,
    "./lightning_logs/mnist_ae_mse/checkpoints/checkpoint.ckpt": ANNAutoencoder,
    "./lightning_logs/cifar10_classifier/checkpoints/epoch=49-step=35150.ckpt": CIFAR10Classifier,
    "./lightning_logs/cifar10_ae_mse/checkpoints/epoch=199-step=35000.ckpt": CIFAR10Autoencoder,
    "./lightning_logs/fmnist_classifier/checkpoints/epoch=49-step=42950.ckpt": MNISTClassifier,
    "./lightning_logs/fmnist_ae_mse/checkpoints/epoch=19-step=8580.ckpt": ANNAutoencoder,
    "./lightning_logs/celeba_classifier/checkpoints/epoch=4-step=11720.ckpt": CelebAClassifier,
    "./lightning_logs/celeba_ae_mse/checkpoints/epoch=49-step=31700.ckpt": CelebAAutoencoder,
    "./lightning_logs/mnist_ccae_mse/checkpoints/checkpoint.ckpt": ClassConstrainedANNAutoencoder
}