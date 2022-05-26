"""
Pytorch-Lightning implementation of the classifier
"""
import sys
sys.path.append("..")

import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from torchmetrics.functional import accuracy
from dataloader import load_mnist, load_cifar
from torch.optim.lr_scheduler import OneCycleLR


class MNISTClassifier(pl.LightningModule):
    """
    This is an implementation of the classifier using ANN
    """
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("######################################")
        # print(x.shape, x.requires_grad)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)

        return loss
    
    def accuracy(self, logits, y):
        # currently IPU poptorch doesn't implicit convert bools to tensor
        # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
        # we can use the accuracy metric.
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        x, y = batch
        
        return self(x)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = self.accuracy(y_hat, y)
        self.log("test_acc", acc)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class CIFAR10Classifier(pl.LightningModule):
    """
    This is an implementation of the CIFAR10 classifier
    """
    def __init__(self, batch_size: int=64, lr: float=0.05) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()

        self.model = torchvision.models.resnet18(pretrained=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        return F.log_softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     # x = x.view(x.size(0), -1)
    #     y_hat = self(x)
    #     loss = self.criterion(y_hat, y)
    #     self.log("train_loss", loss)

    #     return loss
    
    # def accuracy(self, logits, y):
    #     # currently IPU poptorch doesn't implicit convert bools to tensor
    #     # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
    #     # we can use the accuracy metric.
    #     acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
    #     return acc

    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        x, y = batch
        
        return self(x)
    
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     acc = self.accuracy(y_hat, y)
    #     self.log("test_acc", acc)
    #     return acc

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


if __name__ == "__main__":
    
    """
    CIFAR classifier testing
    """
    train_dataloader, valid_dataloader, test_dataloader = load_cifar(
        root="/home/sweta/scratch/datasets/CIFAR/"
    )

    model = CIFAR10Classifier()
    trainer = pl.Trainer(max_epochs=50, gpus=1, default_root_dir="..")
    # trainer.fit(model, train_dataloader, valid_dataloader)  

    model = CIFAR10Classifier.load_from_checkpoint("../lightning_logs/version_10/checkpoints/epoch=49-step=35150.ckpt")
    model.eval()
    preds = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True)
    p = trainer.test(model, dataloaders=test_dataloader)
    print(p)

    """
    MNIST classifier testing
    """
    # train_dataloader, test_dataloader = load_mnist(root="../data/")

    # model = MNISTClassifier()
    # trainer = pl.Trainer(max_epochs=10, gpus=1, default_root_dir="..")
    # # trainer.fit(model, train_dataloader)   

    # model = MNISTClassifier.load_from_checkpoint("../lightning_logs/version_6/checkpoints/epoch=9-step=9370.ckpt")
    # model.eval()
    # # preds = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True)
    # # print(len(preds), preds[0].shape)
    # p = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    # print(p)
