def labelwise_load_mnist(batch_size: int=64, root: str='/scratch/itee/uqsswain/datasets/IMAGENET/', label=7):
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
