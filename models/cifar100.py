import sys

sys.path.append("..")

import torch
import torchvision

import torch.optim as optim

from torch import nn
from termcolor import colored
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.models import (ResNet18_Weights, ResNet50_Weights,
                                ResNet101_Weights)
from dataloader import load_cifar100

def fine_tune(model, last_n):
    for p in model.parameters():
        p.requires_grad = False
    for c in list(model.children())[-last_n:]:
        for p in c.parameters():
            p.requires_grad = True

    return model

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def train(args, model, train_dataloader, valid_dataloader, criterion, optimizer, train_scheduler, warmup_scheduler):
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()  # Set the model to training mode
        
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            print('\r' + 'Epoch [{}/{}]| Batch [{}/{}]| Train Loss: {:.4f}| Train Acc: {:.2f}% '.format(
                epoch+1, args.epochs, batch_idx+1, len(train_dataloader), train_loss/(batch_idx+1), 100.*correct/total
            ), end='', flush=True)
        
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_dataloader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        train_loss /= len(train_dataloader)
        val_loss /= len(valid_dataloader)
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        
        print('\n\n+------------+-------------------+-------------------+----------------+-----------------+-----------------+')
        print('|  Epoch     |  Train Loss       |  Train Accuracy   |  Val Loss      |  Val Accuracy   |  Learning Rate  |')
        print('+------------+-------------------+-------------------+----------------+-----------------+-----------------+')
        print('|  {:2d}        |  {:.6f}           |  {:2.2f}%            |  {:.6f}          |  {:2.2f}%           |  {:.6f}         |'.format(
            epoch+1, train_loss, train_acc, val_loss, val_acc, train_scheduler.get_last_lr()[0]
        ))
        print('+------------+-------------------+-------------------+----------------+-----------------+-----------------+\n')


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.path)
            print(colored(f'Model saved with validation accuracy: {best_val_acc:.2f}%\n', 'green'))
        
        # if args.epochs <= args.warm:
        #     warmup_scheduler.step()
        # else:
        train_scheduler.step()

    return model

if __name__ == "__main__":

    class Args:
        device = "cuda"
        epochs = 300
        batch_size = 128

        # training params
        lr = 0.1
        momentum = 0.9
        weight_decay = 5e-4
        warm = 1
        gamma = 0.1
        # milestones = [60, 120, 160]
        milestones = [150, 225]

        # other params
        infer = True
        root = "/scratch/itee/uqsswain/"
        path = f"{root}artifacts/spaa/classifiers/cifar100/cifar100_resnet101_raw.pt"

    args = Args()

    model = torchvision.models.resnet101(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)

    # model = fine_tune(model, 3)
    model = model.to(args.device)

    # load data
    train_dataloader, valid_dataloader, test_dataloader = load_cifar100(
        batch_size=args.batch_size,
        root=args.root
    )

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    # iter_per_epoch = len(train_dataloader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if not args.infer:
        # Training loop
        model = train(args, model, train_dataloader, valid_dataloader, criterion, optimizer, train_scheduler, None)

    else:
        # code for loading model
        model.load_state_dict(torch.load(args.path))

    # Testing loop
    model.eval()  # set the model to evaluation mode
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_dataloader)
    test_acc = 100. * (test_correct / test_total)

    print('\nTest Loss: {:.6f} | Test Acc: {:.2f}%'.format(test_loss, test_acc))
