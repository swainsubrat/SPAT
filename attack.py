"""
Pytorch implementation of the attack generation
"""
from typing import List
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from dataloader import load_mnist, load_cifar
from utils import save, load
from models import Encoder, Decoder
from classifier import CNN

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print(f"Using {device} as the accelerator")

checkpoint = torch.load('./models/cifar_autoencoder.pth.tar')
print("Found Checkpoint of Autoencoder:)")
encoder = checkpoint["encoder"]
decoder = checkpoint["decoder"]
encoder.to(device)
decoder.to(device)

checkpoint = torch.load('./models/cifar_cnn.pth')
print("Found Checkpoint of Classifier:)")

PATH = './models/cifar_cnn.pth'
classifier = Net()
classifier.load_state_dict(torch.load(PATH))

# classifier = checkpoint["classifier"]
classifier.to(device)

train_dataloader, _, _ = load_cifar(batch_size=1)

attack_item = next(iter(train_dataloader))
attack_data = {
    "image": attack_item[0],
    "label": attack_item[1]
}
save(path="./objects/cifar_attack.pkl", params=attack_data)

attack_data = load(path="./objects/cifar_attack.pkl")
image, label = attack_data["image"], attack_data["label"]
LABEL = label[0]
print(int(LABEL))

image.to(device)
encoded_image = encoder(image)


###########################################
############# ATTACK WITH GA ##############
###########################################


import pygad
import matplotlib.pyplot as plt

# genome = generate_genome(encoded_image.shape[1])
# fitness(genome, encoded_image, decoder, classifier, label)

def fitness(genome: List, genome_idx: List):
    """
    Method to find fitness of a genome/ chromosome
    """
    # perturbation size
    genome = torch.from_numpy(genome).float()
    # print(genome, type(encoded_image))
    recon_1 = decoder(encoded_image)
    recon_2 = decoder(encoded_image + genome)
    f1 = torch.cdist(recon_1.reshape(1, -1), recon_2.reshape(1, -1)).squeeze(dim=1).squeeze(dim=0).detach()
    # print(f"f1: {f1}")

    # semantic distance
    f2 = torch.norm(genome)
    # print(f"f2: {f2}")

    # confidence
    f3 = None
    # classifier.reshape_size = (1, -1, 28, 28)
    probas = classifier(recon_2)
    probs, preds = torch.topk(probas, k=2, dim=1)

    predicted = preds[0][0]
    global LABEL

    # Toggle Targeted and Untargeted
    if predicted == LABEL:
        f3 = - (probs[0][0] - probs[0][1]).detach()
    else:
        f3 = (probs[0][0] - probs[0][1]).detach()
    # print(f"f3: {f3}")
    fit = None
    if predicted == LABEL:
        fit = f3
    else:
        fit = f3 - (0.1 * f1) - (0.3 * f2)

    # fit = f1 + f2 + f3
    fit = float(fit)
    # print(fit)

    return fit

fitness_function = fitness

num_generations = 100
num_parents_mating = 4

sol_per_pop = 5
num_genes = encoded_image.shape[1]

init_range_low = 0
init_range_high = 0.04

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

perturbation = torch.Tensor(solution).reshape(1, -1)
attack_img   = decoder(encoded_image + perturbation)
recons_img   = decoder(encoded_image)

# prediction
probas = classifier(attack_img)
probs, preds = torch.topk(probas, k=2, dim=1)

predicted = preds[0][0]
print(f"Predicted Label: {predicted}")
print(f"Actual Label: {int(LABEL)}")

image      = image.reshape(3, 32, 32)
recons_img = recons_img.reshape(3, 32, 32).detach().numpy()
attack_img = attack_img.reshape(3, 32, 32).detach().numpy()

fig = plt.figure()
columns = 3
rows = 1
fig.add_subplot(rows, columns, 1)
plt.imshow(np.transpose(image, (1, 2, 0)))
fig.add_subplot(rows, columns, 2)
plt.imshow(np.transpose(recons_img, (1, 2, 0)))
fig.add_subplot(rows, columns, 3)
plt.imshow(np.transpose(attack_img, (1, 2, 0)))
plt.savefig(f"./img/cifar_attack_{int(LABEL)}_{predicted}.png", dpi=600)
plt.show()
