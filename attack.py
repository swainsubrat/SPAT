"""
Pytorch implementation of the attack generation
"""
from typing import List
import torch

from dataloader import load_mnist
from utils import save, load
from autoencoder import Encoder, Decoder
from classifier import CNN

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Using {device} as the accelerator")

checkpoint = torch.load('./models/checkpoint_enc_dec.pth.tar')
print("Found Checkpoint :)")
encoder = checkpoint["encoder"]
decoder = checkpoint["decoder"]
encoder.to(device)
decoder.to(device)

checkpoint = torch.load('./models/checkpoint_cnn.pth.tar')
print("Found Checkpoint :)")
classifier = checkpoint["classifier"]
classifier.to(device)

train_dataloader, _ = load_mnist(batch_size=1)

attack_item = next(iter(train_dataloader))
attack_data = {
    "image": attack_item[0],
    "label": attack_item[1]
}
save(path="./objects/attack.pkl", params=attack_data)

attack_data = load(path="./objects/attack.pkl")
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
    f1 = torch.cdist(recon_1, recon_2).squeeze(dim=1).squeeze(dim=0).detach()

    # semantic distance
    f2 = torch.norm(genome)

    # confidence
    f3 = None
    classifier.reshape_size = (1, -1, 28, 28)
    probas = classifier(recon_2)
    probs, preds = torch.topk(probas, k=2, dim=1)

    predicted = preds[0][0]
    global LABEL

    # Toggle Targeted and Untargeted
    if predicted == LABEL:
        f3 = - (probs[0][0] - probs[0][1]).detach()
    else:
        f3 = (probs[0][0] - probs[0][1]).detach()
    
    fit = None
    if predicted == LABEL:
        fit = f3
    else:
        fit = f3 - (0.1 * f1) - (0.3 * f2)

    # fit = f1 + f2 + f3
    fit = float(fit)

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

image      = image.reshape(28, 28)
recons_img = recons_img.reshape(28, 28).detach().numpy()
attack_img = attack_img.reshape(28, 28).detach().numpy()

fig = plt.figure()
columns = 3
rows = 1
fig.add_subplot(rows, columns, 1)
plt.imshow(image)
fig.add_subplot(rows, columns, 2)
plt.imshow(recons_img)
fig.add_subplot(rows, columns, 3)
plt.imshow(attack_img)
plt.savefig(f"./img/attack_{LABEL}.png", dpi=600)
plt.show()
