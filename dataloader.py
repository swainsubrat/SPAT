import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from dataset import CelebaDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_mnist(batch_size: int=64, root: str='/home/sweta/scratch/datasets/MNIST/'):
    """
    Load MNIST data
    """
    t = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: torch.flatten(x))]
                        )
    train = datasets.MNIST(root=root, train=True, download=True, transform=t)
    train_data, valid_data = random_split(train, [55000, 5000])
    test_data  = datasets.MNIST(root=root, train=False, download=True, transform=t)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=4)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, num_workers=4)

    return train_dataloader, valid_dataloader, test_dataloader

def labelwise_load_mnist(batch_size: int=64, root: str='/home/sweta/scratch/datasets/MNIST/', label=7):
    """
    Load MNIST data
    """
    t = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.flatten(x))]
                    )
    train_data = datasets.MNIST(root=root, train=True, download=True, transform=t)
    test_data  = datasets.MNIST(root=root, train=False, download=True, transform=t)

    idx = (train_data.targets!=label)
    train_data.targets = train_data.targets[idx]
    train_data.data    = train_data.data[idx]

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True)

    return train_dataloader, test_dataloader

def load_cifar(batch_size: int=64, root: str="/home/sweta/scratch/datasets/CIFAR10/"):
    """
    Load CIFAR-10 data
    """
    transform_train = transforms.Compose([
        #   transforms.RandomCrop(32, padding = 4),
        #   transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    train_data, valid_data = random_split(train, [45000, 5000])
    test_data  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=4)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, num_workers=4)

    return train_dataloader, valid_dataloader, test_dataloader

def load_celeba(batch_size: int=64, root: str="/home/sweta/scratch/datasets/CelebA/"):
    """
    Load CelebA dataset
    """
    transform_train = transforms.Compose([
                                        transforms.CenterCrop(178),
                                        transforms.Resize(128),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    transform_test = transforms.Compose([
                                        transforms.CenterCrop(178),
                                        transforms.Resize(128),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    train_data = CelebaDataset(txt_path=root+'celeba_gender_attr_train.txt',
                                        img_dir=root+'img_align_celeba/',
                                        transform=transform_train)
    train_data, valid_data = random_split(train_data, [150000, 12079])
    test_data = CelebaDataset(txt_path=root+'celeba_gender_attr_test.txt',
                                    img_dir=root+'img_align_celeba/',
                                    transform=transform_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    return train_dataloader, valid_dataloader, test_dataloader

def load_fashion_mnist(batch_size: int=64, root: str="/home/sweta/scratch/datasets/FashionMNIST/"):
    """
    Load Fashion-MNIST data
    """
    t = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: torch.flatten(x))]
                        )
    train = datasets.FashionMNIST(root=root, train=True, download=True, transform=t)
    train_data, valid_data = random_split(train, [55000, 5000])
    test_data  = datasets.FashionMNIST(root=root, train=False, download=True, transform=t)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=4)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, num_workers=4)

    return train_dataloader, valid_dataloader, test_dataloader

DATALOADER_MAPPINGS = {
    "mnist": load_mnist,
    "cifar10": load_cifar,
    "celeba": load_celeba,
    "fmnist": load_fashion_mnist
}

if __name__ == "__main__":
    train_dataloader, test_dataloader = load_celeba()
    print(len(train_dataloader))
    for x, y in train_dataloader:
        print(x[0])
        break
    