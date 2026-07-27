import numpy as np
import torch
import torch.nn as nn

class KappaCNN(nn.Module):

    def __init__(self, feat_dimension = 32):
        super().__init__()
        self.conventional = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding = 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding = 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding = 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(32, feat_dimension)

    def forward(self, x):
        h = self.conventional(x).flatten(1)

        return self.fc(h)
    

class KappaCNNForecaster:
    def __init__(self, feat_dimension = 32, hidden = 32, lr = 1e-3, epochs = 100, device = "cpu"):

        self.feat_dimension = feat_dimension
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.cnn = None
        self.gru = None
        self.head = None


    def _group_by_ar(self, records):

        by_ar = {}
        for record in records:
            by_ar.setdefault(record["ar_id"], []).append(record)
        
        for ar in by_ar:
            by_ar[ar].sort(key = lambda record: record["timestamp"])

        
        return by_ar
    
    def _forward_ar(self, records):
        maps = np.stack([record["map"] for record in records]).astype(np.float32)
        x = torch.tensor(maps).unsqueeze(1).to(self.device)
        feats = self.cnn(x)
        out, _ = self.gru(feats.unsqueeze(0))

        return self.head(out).reshape(-1)