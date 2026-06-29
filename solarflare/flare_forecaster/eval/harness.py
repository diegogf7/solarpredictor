
from .metrics import tss, hss, bss, expected_calibration_error
from .splits import apply_splits, no_overlap
import numpy as np



def evaluate(model, train_records, val_records, test_records):

    no_overlap(train_records, val_records, test_records)

    model.fit(train_records)

    probabilities = model.predict_probability(test_records)

    labels = [k["label"] for k in test_records]

    threshold = 0.5

    predictions = [1 if p >= threshold else 0 for p in probabilities]

    return {
        "tss": tss(labels, predictions),
        "hss": hss(labels, predictions),
        "bss": bss(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities),
        "n_test": len(labels),
        "n_positive": sum(labels)


    }