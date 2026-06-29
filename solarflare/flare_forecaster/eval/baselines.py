from sklearn.linear_model import LogisticRegression
import numpy as np


class Baseline:

    def fit(self, record):

        self._last_label = {}

        for r in sorted(record, key = lambda r: r["timestamp"]):
            self._last_label[r["ar_id"]] = float(r["label"])

    def predict_probability(self, record):

        return [self._last_label.get(k["ar_id"], 0.0) for k in record]

class ClimateBaseline:

    def fit(self, records):

        labels = [r["label"] for r in records]

        if labels:

            self._rate = sum(labels) / len(labels)
        else:

            self._rate = 0.0

    def predict_probability(self, records):

        return [self._rate for _ in records]
    

class ScalarBaseline:

    def fit(self, record):

        X = np.array([k["features"] for k in record])
        y = np.array([k["label"] for k in record])

        self._model = LogisticRegression(max_iter = 2000).fit(X, y)

    def predict_probability(self, record):

        X = np.array([k["features"] for k in record])
        return self._model.predict_proba(X)[:, 1].tolist()