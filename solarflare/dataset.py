import os 
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class SolarFlareDataset(Dataset):
    def __init__(self, csv_file, images_direction, transform = None):
        self.images_direction = images_direction
        self.transform = transform
        #self.samples = [] #we have the filenames and the labels here

        df = pd.read_csv(csv_file)

        self.samples = list(zip(df['filename'], df['class']))

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