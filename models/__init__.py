from models.autoencoder import (ANNAutoencoder, ClassConstrainedANNAutoencoder,
                                CelebAAutoencoder, CIFAR10Autoencoder, CIFAR10VAE,
                                CIFAR10LightningAutoencoder, MNISTCNNAutoencoder,
                                CIFAR10NoisyLightningAutoencoder, ImagenetAutoencoder)
from models.classifier import (MNISTClassifier, CelebAClassifier, CIFAR10Classifier,
                               MNISTCNNClassifier, ImagenetClassifier)

MODEL_MAPPINGS = {
    "./lightning_logs/mnist_classifier/checkpoints/epoch=9-step=9370.ckpt": MNISTClassifier,
    "./lightning_logs/mnist_ae_mse/checkpoints/checkpoint.ckpt": ANNAutoencoder,
    "./lightning_logs/cifar10_classifier/checkpoints/epoch=49-step=35150.ckpt": CIFAR10Classifier,
    "./lightning_logs/cifar10_ae_mse/checkpoints/epoch=199-step=35000.ckpt": CIFAR10Autoencoder,
    "./lightning_logs/fmnist_classifier/checkpoints/epoch=49-step=42950.ckpt": MNISTClassifier,
    "./lightning_logs/fmnist_ae_mse/checkpoints/epoch=19-step=8580.ckpt": ANNAutoencoder,
    "./lightning_logs/celeba_classifier/checkpoints/epoch=4-step=11720.ckpt": CelebAClassifier,
    "./lightning_logs/celeba_ae_mse/checkpoints/epoch=49-step=31700.ckpt": CelebAAutoencoder,
    "./lightning_logs/mnist_ccae_mse/checkpoints/checkpoint.ckpt": ClassConstrainedANNAutoencoder,
    "./lightning_logs/version_39/checkpoints/epoch=499-step=175500.ckpt": CIFAR10VAE,
    "./lightning_logs/cifar10_ae_nll/checkpoints/epoch=199-step=70200.ckpt": CIFAR10LightningAutoencoder,
    "./lightning_logs/mnist_cnn_classifier/checkpoints/epoch=49-step=42950.ckpt": MNISTCNNClassifier,
    "./lightning_logs/mnist_cnn_ae/checkpoints/epoch=19-step=8580.ckpt": MNISTCNNAutoencoder,
    "./lightning_logs/version_39/checkpoints/epoch=199-step=70200.ckpt": CIFAR10NoisyLightningAutoencoder,
    "none": ImagenetClassifier,
    "/home/sweta/scratch/models/imagenet-vgg16.pth": ImagenetAutoencoder
}