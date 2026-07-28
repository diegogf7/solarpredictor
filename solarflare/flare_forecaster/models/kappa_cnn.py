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
    
    def fit(self, records):
        self.cnn = KappaCNN(self.feat_dimension).to(self.device)
        self.gru = nn.GRU(self.feat_dimension, self.hidden, batch_first = True).to(self.device)
        self.head = nn.Linear(self.hidden, 1).to(self.device)

        parameters = list(self.cnn.parameters()) + list(self.gru.parameters()) + list(self.head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr = self.lr)
        loss_function = nn.BCEWithLogitsLoss()

        groups = list(self._group_by_ar(records).values())
        for epoch in range(self.epochs):
            total = 0.0

            for recs in groups:
                y = torch.tensor([float(record["label"]) for record in recs]).to(self.device)
                optimizer.zero_grad()

                logits = self._forward_ar(recs)
                loss = loss_function(logits, y)
                loss.backward()
                optimizer.step()

                total += loss.item()
            if epoch % 25 == 0:
                print(f"epoch {epoch: 3d} | loss {total / len(groups):.4f}")

        return self

    def predict_probability(self, records):
        probability_by_id = {}

        with torch.no_grad():

            for recs in self._group_by_ar(records).values():

                probability = torch.sigmoid(self._forward_ar(recs)).cpu().numpy()
                for r, p in zip(recs, probability):
                    probability_by_id[id(r)] = float(p)

        return [probability_by_id[id(r)] for r in records]