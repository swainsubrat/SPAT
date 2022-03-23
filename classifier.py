"""
Pytorch implementation of the Autoencoder
"""
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from torch import nn
from utils import accuracy
from dataloader import load_mnist
	
def save_checkpoint(epoch, classifier, optimizer, path='./models/ciphar_cnn.pth.tar'):
    state = {'epoch': epoch,
             'classifier': classifier,
             'optimizer': optimizer}

    filename = path
    torch.save(state, filename)


class CNN(nn.Module):
    """
    Encoder to encode the image into a hidden state
    """
    def __init__(self, reshape_size):
        super().__init__()
        self.reshape_size = reshape_size

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.Mish(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5),
            nn.Mish(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 50),
            nn.Mish(),
            nn.Linear(50, 10),
            nn.LogSoftmax()
        )
    
    def forward(self, images):
        images = torch.reshape(images, self.reshape_size)
        x = self.conv(images)
        x = x.reshape(x.shape[0], -1)
        out = self.classifier(x)
        
        return out

if __name__ == "__main__":
    batch_size = 64
    
    print(f"Using {device} as the accelerator")
    train_dataloader, test_dataloader = load_mnist(batch_size=batch_size, root='./data/')
    try:
        # try loading checkpoint
        checkpoint = torch.load('./models/ciphar_cnn.pth.tar')
        print("Found Checkpoint :)")
        classifier = checkpoint["classifier"]
        classifier.to(device)

    except:
        # train the model from scratch
        print("Couldn't find checkpoint :(")

        classifier = CNN(reshape_size=(batch_size, -1, 28, 28))
        classifier.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-5)

        for epoch in range(10):
            for i, (image, label) in enumerate(train_dataloader):
                image.to(device)
                out = classifier(image)
                
                loss = criterion(out, label)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if i % 100 == 0 and i != 0:
                    print(f"Epoch: [{epoch+1}][{i}/{len(train_dataloader)}] Loss: {loss.item(): .4f}")

            save_checkpoint(epoch, classifier, optimizer)
    
    y_hat, y = [], []
    for i, (image, label) in enumerate(test_dataloader):
        image.to(device)
        out = classifier(image)
        prob, idxs = torch.max(out, dim=1)

        y.extend(label.tolist())
        y_hat.extend(idxs.tolist())

    print(accuracy(y, y_hat))
