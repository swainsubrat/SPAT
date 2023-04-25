import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImagenetDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, inverse_label_path, img_dir, transform=None):
    
        self.img_names = os.listdir(img_dir)
        self.img_dir = img_dir
        self.cat_to_label= eval(open(inverse_label_path,"r").read())
        self.transform = transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                    self.img_names[index]))
        
        cat_name = " ".join(self.img_names[index].split("_")[1:]).split(".")[:-1][0]
        label = self.cat_to_label[cat_name]
        if self.transform is not None:
            img = self.transform(img)
        
        if img.shape == (1, 224, 224):
            img = torch.cat([img]*3, dim=0)
        
        # img = self.normalize(img) # for normalization after ensuring shape (3, 224, 224)

        return img, label

    def __len__(self):
        return len(self.img_names)


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, txt_path, img_dir, transform=None):
    
        df = pd.read_csv(txt_path, sep=" ", index_col=0)
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.img_names = df.index.values
        self.y = df['Male'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                    self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]


if __name__ == "__main__":
    img_dir = "/home/harsh/scratch/datasets/IMAGENET/images/"
    inverse_label_path = "/home/harsh/scratch/datasets/IMAGENET/labels/inverse_labels.txt"
    obj = ImagenetDataset(
        inverse_label_path=inverse_label_path,
        img_dir=img_dir
    )
    for i in range(1, 10):
        img, label = obj.__getitem__(i)
        print(img, label)
    # txt_file = open("/home/harsh/scratch/datasets/IMAGENET/labels/labels.txt","r").read()
    # _dict = eval(txt_file)
    # label_dict = {}
    # for key, values in _dict.items():
    #     values = values.split(",")
    #     for value in values:
    #         label_dict[value.strip()] = key

    # f = open("/home/harsh/scratch/datasets/IMAGENET/labels/inverse_labels.txt","w")
    # f.write( str(label_dict) )
    # f.close()