import numpy as np

def binary_confusion(y_true, y_prediction):

    #using a binary metric where 1 is true and 0 is false
    #for making a confusion matrix
    y_true = np.asarray(y_true).astype(int)
    y_prediction = np.asarray(y_prediction).astype(int)
    tp = int(np.sum((y_prediction == 1) & (y_true == 1)))

    fp = int(np.sum((y_prediction == 1) & (y_true == 0)))
    tn = int(np.sum((y_prediction == 0) & (y_true == 0)))

    fn = int(np.sum((y_prediction == 0) & (y_true == 1)))

    return tp, fp, tn, fn

def tss(y_true, y_prediction):
    #getting the true skill statistics
    tp, fp, tn, fn = binary_confusion(y_true, y_prediction)

    #defining a variable to determine how good our classification is
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return recall - false_positive_rate

def hss(y_true, y_prediction):

    #getting the Heidike Skill score
    tp, fp, tn, fn = binary_confusion(y_true, y_prediction)

    num = 2.0 * (tp * tn - fp * fn)

    density = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return num / density if density else 0.0

def brier_score(y_true, y_probability):
    y_true = np.asarray(y_true, dtype = float)

    y_probability = np.asarray(y_probability, dtype = float)

    return float(np.mean((y_probability - y_true) ** 2))

def bss(y_true, y_probability, climatology = None):

    #comparing the brier skill score to the climatology reference
    #note here that 1 is perfect and 0 is just you randomly predicting a value

    y_true = np.asarray(y_true, dtype = float)
    if climatology is None:
        climatology = float(np.mean(y_true))

    brier = brier_score(y_true, y_probability)
    brier_reference = brier_score(y_true, np.full_like(y_true, climatology))

    return 1.0 - brier / brier_reference if brier_reference else 0.0

    