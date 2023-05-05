"""
Pytorch-Lightning implementation of the classifier
"""
import sys

sys.path.append("..")

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import logit, nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from torchvision.models import Inception_V3_Weights, inception_v3

from dataloader import (load_celeba, load_cifar, load_fashion_mnist,
                        load_imagenet_x, load_mnist, load_gtsrb)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MNISTClassifier(pl.LightningModule):
    """
    This is an implementation of the classifier using ANN
    """
    def __init__(self, lr: float=1e-3, batch_size: int=64) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def accuracy(self, logits, y):
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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = self.accuracy(y_hat, y)
        self.log("valid_acc", acc)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class MNISTCNNClassifier(pl.LightningModule):
    """
    This is an implementation of the classifier using CNN
    """
    def __init__(self, lr: float=1e-3, momentum=0.5, batch_size=64) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.lr         = lr
        self.momentum   = momentum
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),

            nn.Linear(1024, 200),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((-1, 1, 28, 28))
        out = self.model(x)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def accuracy(self, logits, y):
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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = self.accuracy(y_hat, y)
        self.log("valid_acc", acc)
        return acc

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=5e-4,
        )
        steps_per_epoch = 55000 // self.batch_size
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

    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        x, y = batch
        
        return self(x)

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


class ImagenetClassifier(pl.LightningModule):
    """
    This is an implementation of the CIFAR10 classifier
    """
    def __init__(self, batch_size: int=32, lr: float=0.05) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.batch_size = batch_size
        # self.criterion = nn.CrossEntropyLoss()

        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        return F.log_softmax(x, dim=1)

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        # import pdb; pdb.set_trace()
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=1000)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        x, y = batch
        
        return self(x)


class CelebAClassifier(pl.LightningModule):
    """
    Implementation of CelebA gender classification
    """
    def __init__(self, batch_size: int=64, lr: float=0.001) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.aux_logits = False
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()

        self.model = torchvision.models.resnet50(weights=None)
        if self.aux_logits:
            self.model.AuxLogits.fc = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
        self.model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=1), nn.Sigmoid())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.aux_logits:
            x, aux_x = self.model(x)
            return x.flatten(), aux_x.flatten()
        else:
            x = self.model(x)
            return x.flatten()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.float32)
        if self.aux_logits:
            logits, aux_logits = self(x)
            loss = F.binary_cross_entropy(logits, y) + 0.4 * F.binary_cross_entropy(aux_logits, y)
        else:
            logits = self(x)
            loss = F.binary_cross_entropy(logits, y)

        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        y = y.to(torch.float32)
        if self.aux_logits:
            logits, aux_logits = self(x)
            loss = F.binary_cross_entropy(logits, y) + 0.4 * F.binary_cross_entropy(aux_logits, y)
        else:
            logits = self(x)
            loss = F.binary_cross_entropy(logits, y)

        preds = (logits>0.5).int()
        acc = accuracy(preds, y, task="binary")

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.float32)
        logits = self.model(x).flatten()
        loss = F.binary_cross_entropy(logits, y)
        # import pdb; pdb.set_trace()
        preds = (logits>0.5).int()
        y = y.int()
        acc = accuracy(preds, y, task="binary")

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        
        return pred.flatten()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class GTSRBClassifier(pl.LightningModule):
    """
    Implementation of GTSRB classification
    """
    def __init__(self, batch_size: int=64, lr: float=0.001) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.aux_logits = False
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()

        self.model = torchvision.models.resnet34(weights=None)
        if self.aux_logits:
            self.model.AuxLogits.fc = nn.Sequential(nn.Linear(768, 1))
        self.model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=43))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.aux_logits:
            x, aux_x = self.model(x)
            return x.flatten(), aux_x.flatten()
        else:
            x = self.model(x)
            return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = y.to(torch.float32)
        if self.aux_logits:
            logits, aux_logits = self(x)
            loss = F.cross_entropy(logits, y) + 0.4 * F.cross_entropy(aux_logits, y)
        else:
            logits = self(x)
            loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        # y = y.to(torch.float32)
        if self.aux_logits:
            logits, aux_logits = self(x)
            loss = F.cross_entropy(logits, y) + 0.4 * F.cross_entropy(aux_logits, y)
        else:
            logits = self(x)
            loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=43)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
        # x, y = batch
        # # y = y.to(torch.float32)
        # logits = self(x)
        # loss = F.cross_entropy(logits, y)
        # # import pdb; pdb.set_trace()
        # preds = torch.argmax(logits, dim=1)
        # # y = y.int()
        # acc = accuracy(preds, y, task="multiclass", num_classes=43)

        # self.log("test_loss", loss, prog_bar=True)
        # self.log("test_acc", acc, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        
        return pred.flatten()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":

    """
    GTSRB classifier testing
    """
    train_dataloader, valid_dataloader, test_dataloader = load_gtsrb(
        batch_size=128,
        root="/scratch/itee/uqsswain/"
    )

    model = GTSRBClassifier()
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu",
        devices=1,
        default_root_dir="/scratch/itee/uqsswain/artifacts/spaa/classifiers/gtsrb/"
    )
    trainer.fit(model, train_dataloader, valid_dataloader)

    # model = GTSRBClassifier.load_from_checkpoint(
    #     "/scratch/itee/uqsswain/artifacts/spaa/classifiers/gtsrb/lightning_logs/version_546423/checkpoints/epoch=9-step=1700.ckpt"
    # )
    # model.eval()
    # preds = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True)
    # print(len(preds), preds[0].shape)
    p = trainer.test(model, dataloaders=test_dataloader, verbose=True)
    print(p)

    """
    Imagenet classifier testing
    """
    # model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(torch.device("cpu"))
    # model.eval()

    # # Testing
    # train_dataloader = load_imagenet_x(
    #     root="/home/harsh/scratch/datasets/IMAGENET/", batch_size=32
    # )
    # images, labels = next(iter(train_dataloader))
    # preds = model(images)

    # # Testing lighting model
    # model = ImagenetClassifier()
    # model.eval()
    # trainer = pl.Trainer(max_epochs=50, accelerator="cpu", devices=1, default_root_dir="..", enable_checkpointing=False) 
    # p = trainer.test(model, dataloaders=train_dataloader)
    # print(p)
    """
    MNIST CNN classifier testing
    """
    # train_dataloader, valid_dataloader, test_dataloader = load_mnist(root="/home/sweta/scratch/datasets/MNIST/")

    # model = MNISTCNNClassifier()
    # trainer = pl.Trainer(max_epochs=50, gpus=1, default_root_dir="..")
    # # trainer.fit(model, train_dataloader, valid_dataloader)

    # model = MNISTCNNClassifier.load_from_checkpoint("../lightning_logs/mnist_cnn_classifier/checkpoints/epoch=49-step=42950.ckpt")
    # model.eval()
    # p = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    # print(p)
    """
    CIFAR classifier testing
    """
    # train_dataloader, valid_dataloader, test_dataloader = load_cifar()

    # model = CIFAR10Classifier()
    # trainer = pl.Trainer(max_epochs=50, gpus=1, default_root_dir="..", enable_checkpointing=False)
    # # trainer.fit(model, train_dataloader, valid_dataloader)  

    # model = CIFAR10Classifier.load_from_checkpoint("../lightning_logs/cifar10_classifier/checkpoints/epoch=49-step=35150.ckpt")
    # model.eval()
    # # preds = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True)
    # p = trainer.test(model, dataloaders=test_dataloader)
    # print(p)

    """
    MNIST classifier testing
    """
    # train_dataloader, valid_dataloader, test_dataloader = load_mnist(root="/home/sweta/scratch/datasets/MNIST/")

    # model = MNISTClassifier()
    # trainer = pl.Trainer(max_epochs=10, gpus=1, default_root_dir="..")
    # # trainer.fit(model, train_dataloader)   

    # model = MNISTClassifier.load_from_checkpoint("../lightning_logs/mnist_classifier/checkpoints/epoch=9-step=9370.ckpt")
    # model.eval()
    # # preds = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True)
    # # print(len(preds), preds[0].shape)
    # p = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    # print(p)

    """
    CelebA classifier testing
    """
    # train_dataloader, valid_dataloader, test_dataloader = load_celeba(
    #     batch_size=128,
    #     root="/scratch/itee/uqsswain/"
    # )

    # model = CelebAClassifier()
    # trainer = pl.Trainer(
    #     max_epochs=10,
    #     accelerator="gpu",
    #     devices=1,
    #     default_root_dir="/scratch/itee/uqsswain/artifacts/spaa/classifiers/celeba/"
    # )
    # trainer.fit(model, train_dataloader, valid_dataloader)

    # # model = CelebAClassifier.load_from_checkpoint("../lightning_logs/celeba_classifier/checkpoints/epoch=4-step=11720.ckpt")
    # model.eval()
    # # preds = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True)
    # # print(len(preds), preds[0].shape)
    # p = trainer.test(model, dataloaders=test_dataloader, verbose=True)
    # print(p)