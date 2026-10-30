"""Dataset that gives skin lesion images and their labels to PyTorch."""

import torch
from PIL import Image
from torch.utils.data import Dataset

class LesionDataset(Dataset):
    """Return (image, label) pairs from a split table. Can be train.csv, val.csv, or test.csv"""
    def __init__(self, df, transform):
        self.paths = df["path"].tolist() # image file
        self.labels = df["label"].tolist() # 1 malignant or 0 benign 
        self.transform = transform # transformation for an image

    def __len__(self):
        # mumber of images
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB") # open image i from disk in rgb format
        label = torch.tensor(self.labels[i], dtype=torch.float32) # label as a decimal number
        return self.transform(image), label # run transform every time an image is loaded