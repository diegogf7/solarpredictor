import numpy as np
import torch
import torch.nn as nn

class TemporalForecaster:

    def __init__(self, n_features = 4, hidden = 32, lr = 1e-3, epochs = 150, device = "cpu"):
        self.n_features = n_features
        self.hidden = hidden
        self.lr =lr
        self.epochs = epochs
        self.device = device
        self.head = None
        self.model = None
        self.mean = None
        self.std = None



    def _group_by_ar(self, records):

        by_ar = {}
        for record in records:
            by_ar.setdefault(record["ar_id"], []).append(record)

        for ar in by_ar:
            by_ar[ar].sort(key = lambda record: record["timestamp"])

        
        return by_ar
    
    def fit(self, records):
        feats = np.array([record["features"] for record in records], dtype = np.float64)

        self.mean = feats.mean(axis =0)
        self.std = feats.std(axis = 0) + 1e-8

        self.model = nn.GRU(self.n_features, self.hidden, batch_first = True).to(self.device)
        self.head = nn.Linear(self.hidden, 1).to(self.device)

        parameters = list(self.model.parameters()) + list(self.head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr = self.lr)

        loss_function = nn.BCEWithLogitsLoss()

        sequences = []

        for ar, recs in self._group_by_ar(records).items():

            x = ((np.array([r["features"] for r in recs], dtype = np.float64) - self.mean) / self.std).astype(np.float32)
            y = np.array([r["label"] for r in recs], dtype = np.float32)

            sequences.append((torch.tensor(x), torch.tensor(y)))

        for epoch in range(self.epochs):

            total_loss = 0.0

            for x, y in sequences:
                x = x.unsqueeze(0).to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()

                out, _ = self.model(x)

                logits = self.head(out).reshape(-1)

                loss = loss_function(logits, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 50 == 0:
                print(f"epoch {epoch:3d} with loss {total_loss / len(sequences):.5f}")
        
        return self
    
    def predict_probability(self, records):

        probability_by_id = {}

        with torch.no_grad():
            for ar, recs in self._group_by_ar(records).items():
                x = ((np.array([r["features"] for r in recs], dtype = np.float64) - self.mean) / self.std).astype(np.float32)
                x = torch.tensor(x).unsqueeze(0).to(self.device)
                out, _ = self.model(x)
                probability = torch.sigmoid(self.head(out).reshape(-1)).cpu().numpy()

                for r, prob in zip(recs, probability):
                    probability_by_id[id(r)] = float(prob)

        return [probability_by_id[id(r)] for r in records]
