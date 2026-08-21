import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SolarFlareDataset, transform
from model import SolarCNN

device = torch.device('mps')
