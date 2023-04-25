# Semantic-Preserving-Adversarial-Attack

The task is to perform attack on 3: MNIST, CIFAR10, and CELEBA Dataset.

What do I need for the attack?
1. Autoencoder tranined on the dataset
2. A classifier to fool
3. Attack scores on the prior techniques and new techniques

File location:

The code is written in such a way that, all the things except the files (models, objects, datasets) are kept in the project artifact folder. Here, in the code, we've to mention the location of the corresponding artifact folder. Once, we've specified it, the others will follow the path as:

```
project_artifact\
    datasets\ # dataset specific to the project or experiment
    checkpoints\ # models of the project
    objects\ # objects saved from the project
```

