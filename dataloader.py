import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from dataset import CelebaDataset, ImagenetDataset
from torchvision.models import Inception_V3_Weights

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def my_collate_fn(data):
    # TODO: Implement your function
    # But I guess in your case it should be:
    return tuple(data)
    # return data

def lambda_function(x):
    return torch.flatten(x)

def load_mnist(batch_size: int=64, root: str='~/scratch/datasets/MNIST/'):
    """
    Load MNIST data
    """
    t = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda_function)]
                        )
    train = datasets.MNIST(root=root, train=True, download=True, transform=t)
    train_data, valid_data = random_split(train, [55000, 5000])
    test_data  = datasets.MNIST(root=root, train=False, download=True, transform=t)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=4)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, num_workers=4)

    return train_dataloader, valid_dataloader, test_dataloader

def labelwise_load_mnist(batch_size: int=64, root: str='~/scratch/datasets/MNIST/', label=7):
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

def load_mnist_x(batch_size: int=64, root: str="~/scratch/datasets/CIFAR10/", dataset_len=1000):
    """
    Load MNIST 1000 data
    """
    torch.manual_seed(43)
    t = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda_function)]
                        )
    
    train = datasets.MNIST(root=root, train=True, download=True, transform=t)
    train_data, _ = random_split(train, [dataset_len, len(train)-dataset_len])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, pin_memory=True, num_workers=4)

    return train_dataloader

def load_cifar(batch_size: int=64, root: str="~/scratch/datasets/CIFAR10/"):
    """
    Load CIFAR-10 data
    """
    transform_train = transforms.Compose([
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

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=4)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, num_workers=4)

    return train_dataloader, valid_dataloader, test_dataloader

def load_cifar_x(batch_size: int=64, root: str="~/scratch/datasets/CIFAR10/", dataset_len=1000):
    """
    Load CIFAR-10 1000 data
    """
    torch.manual_seed(43)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    train_data, _ = random_split(train, [dataset_len, len(train)-dataset_len])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, pin_memory=True, num_workers=4)

    return train_dataloader

def load_celeba(batch_size: int=64, root: str="~/scratch/datasets/CelebA/"):
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

def load_fashion_mnist(batch_size: int=64, root: str="~/scratch/datasets/FashionMNIST/"):
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

def load_gtsrb(batch_size: int=64, root: str="/home/sweta/scratch/datasets/GTSRB/"):
    """
    Load GTSRB data
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))
        ])
    
    train = datasets.GTSRB(root=root, split="train", download=True, transform=transform)
    train_data, valid_data = random_split(train, [len(train)-5000, 5000])
    test_data  = datasets.GTSRB(root=root, split="test", download=True, transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=4)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, num_workers=4)

    return train_dataloader, valid_dataloader, test_dataloader

def load_imagenet(batch_size: int=32, root: str="/scratch/itee/uqsswain/datasets/IMAGENET"):
    """
    Load Imagenet data
    """
    weights = Inception_V3_Weights.DEFAULT
    # transform = weights.transforms()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_data = ImagenetDataset(
        inverse_label_path="/scratch/itee/uqsswain/datasets/IMAGENET/labels/inverse_labels.txt",
        img_dir="/scratch/itee/uqsswain/datasets/IMAGENET/images/",
        transform=transform
    )

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=False, num_workers=0)

    return train_dataloader


DATALOADER_MAPPINGS = {
    "mnist": load_mnist,
    "cifar10": load_cifar,
    "celeba": load_celeba,
    "fmnist": load_fashion_mnist,
    "gtsrb": load_gtsrb,
    "cifar101000": load_cifar_x,
    "mnist1000": load_mnist_x,
    "imagenet1000": load_imagenet
}

if __name__ == "__main__":
    train_dataloader = load_imagenet(batch_size=1)
    for imgs, labels in train_dataloader:
        print(imgs.shape)
        print(labels)
        break
