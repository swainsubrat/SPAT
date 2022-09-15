from models.autoencoder import (ANNAutoencoder, ClassConstrainedANNAutoencoder,
                                CelebAAutoencoder, CIFAR10Autoencoder)
from models.classifier import (MNISTClassifier, CelebAClassifier, CIFAR10Classifier)

MODEL_MAPPINGS = {
    "./lightning_logs/version_6/checkpoints/epoch=9-step=9370.ckpt": MNISTClassifier,
    "./lightning_logs/version_0/checkpoints/epoch=9-step=9370.ckpt": ANNAutoencoder
}