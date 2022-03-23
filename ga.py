import torch

from typing import List

from autoencoder import Decoder
from classifier import CNN

Genome = torch.Tensor
Population = List[Genome]


def generate_genome(length: int) -> Genome:
    return torch.rand(length)


def population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: Genome, encoded_image: torch.Tensor, decoder: Decoder, classifier: CNN, label: torch.Tensor):
    """
    Method to find fitness of a genome/ chromosome
    """
    # perturbation size
    recon_1 = decoder(encoded_image)
    recon_2 = decoder(encoded_image + genome)
    f1 = torch.cdist(recon_1, recon_2).squeeze(dim=1).squeeze(dim=0).detach()

    # semantic distance
    f2 = torch.norm(genome)

    # confidence
    f3 = None
    classifier.reshape_size = (1, -1, 28, 28)
    probas = classifier(recon_2)
    probs, preds = torch.topk(probas, k=2, dim=1)

    predicted = preds[0][0]
    label     = label[0]

    if predicted == label:
        f3 = - (probs[0][0] - probs[0][1]).detach()
    else:
        f3 = (probs[0][0] - probs[0][1]).detach()
    
    fit = f1 + f2 + f3
    
    return fit

if __name__ == "__main__":
    print(population(5, 5))