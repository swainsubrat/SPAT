import torch
from tqdm import tqdm
from torchvision.models import inception_v3, Inception_V3_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from dataloader import load_imagenet

train_dataloader = load_imagenet()
model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
print("hello")
model.eval()

for images, labels in tqdm(train_dataloader):
    images, labels = images.to(device), labels.to(device)
    preds = model(images)
    print("Done!!")