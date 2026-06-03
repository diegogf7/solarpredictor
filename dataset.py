import os 
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class SolarFlareDataset(Dataset):
    def __init__(self, label_file, images_direction, transform = None):
        self.images_direction = images_direction
        self.transform = transform
        #self.samples = [] #we have the filenames and the labels here

        df = pd.read_csv(label_file, header=None, names = ['filename', 'flare_class'])
        df['label'] = df['flare_class'].apply(
            lambda x: 1 if x[0] == 'M' or x[0] == "X" else 0
        )

        self.samples = list(zip(df['filename'], df['label']))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        image_path = os.path.join(self.images_direction, filename)
        image = Image.open(image_path).convert('L') #need the grayscale

        if self.transform:
            image = self.transform(image)

        
        return image, torch.tensor(label, dtype = torch.float32)


transform = transforms.Compose([
    transforms.ToTensor(), #changing from our PIL (just basic python) to our tensor image for PyTorch
    transforms.Normalize(mean= [0.5], std = [0.5]) #need to shift to our -1 to 1 range
])