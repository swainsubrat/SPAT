import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_mnist(batch_size: int=64, root: str='./data/'):
    """
    Load MNIST data
    """
    t = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Lambda(lambda x: torch.flatten(x))]
                       )
    train_data = datasets.MNIST(root=root, train=True, download=True, transform=t)
    test_data  = datasets.MNIST(root=root, train=False, download=True, transform=t)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_dataloader, test_dataloader


def load_cifar(batch_size: int=64, root: str="./data/"):
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

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

def load_celeba(batch_size: int=64, root: str="./data/"):
    """
    Load CelebA dataset
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    train_data = datasets.CelebA(root=root, split='train', download=True, target_type="identity", transform=transform_train)
    valid_data = datasets.CelebA(root=root, split='valid', download=True, target_type="identity", transform=transform_train)
    test_data = datasets.CelebA(root=root, split='test', download=True, target_type="identity", transform=transform_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataloader, valid_dataloader, test_dataloader = load_celeba(root='./data/')
    print(train_dataloader.shape)
    for x, y in train_dataloader:
        print(x[0])
        print(y[0])
        break