# Semantic-Preserving-Adversarial-Attack

The task is to perform attack on 3: CIFAR10, GTSRB and MNIST dataset.

What do I need for the attack?
1. Autoencoder tranined on the dataset
2. A classifier to fool
3. Attack Success Rates and LPIPS on the prior techniques and new techniques

## File location:

The code is written in such a way that, all the things except the files (models, objects, datasets) are kept in the project artifact folder. Here, in the code, we've to mention the location of the corresponding artifact folder. Once, we've specified it, the others will follow the path as:

```
project_artifact\
    datasets\ # dataset specific to the project or experiment
    checkpoints\ # models of the project
    objects\ # objects saved from the project
```

## Running an attack (X and SPAT-X)
- To run and get the results for any attack X and SPAT-X run the following code:
```
python attack_main.py --model-name cifar10_cnn_1 --ae-name ann_128 --dataset-len 1000 --attack-name pgd --batch-size 32 --eval
```
Here:
1. Classifier name is ```cifar10_cnn_1```. For choosing different models, refer the file ```configs/dataset_name.yml``` in the ```classifiers``` section.
2. Autoencoder name is ```ann_128```. For choosing different models, refer the file ```configs/dataset_name.yml``` in the ```autoencoders``` section.
3. Dataset length is 1000, you can choose anywhere between 1 to 10,000.
4. Attack name is ```pgd```. Choose one of: ```fgsm, bim, pgd, cnw, deepfool, elastic```. Here ```elastic``` corresponds to EAD and ```deepfool``` corresponds to DF in the paper.
5. Batch size is 32

For other parametes and flags, consider checking the ```attack_main.py```

## Adding a dataset:
- Create a datoloader and put it in the dataloader mapping dict in the file ```dataloader.py```
- Create a classifier in the file ```models/classifier.py```
- Create an autoencoder in the file ```models/autoencoder.py```
- Create a config file in the folder ```configs/``` as ```dataset_name.yml```
- Update the ```dataset_name.yml``` using the classifier checkpoint path and autoencoder checkpoint path.
- Update the same in the file ```models/__init__.py```
- Run ```attack_main.py``` with appropiate command
