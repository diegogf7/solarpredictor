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

def expected_calibration_error(y_true, y_probability, n_bins = 10):

    #probability_true, probability_prediction = calibration_curve(y_true, y_probability, n_bins = n_bins, strategy = "uniform")
    
    y_true = np.asarray(y_true, dtype = float)
    y_probability = np.asarray(y_probability, dtype = float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_probability, bin_edges, right = True) - 1

    #need to make sure we only get positive values for the ids

    bin_ids = np.clip(bin_ids, 0, n_bins - 1)


    bin_totals = np.bincount(bin_ids, minlength = n_bins) 

    nonzero = bin_totals > 0

    ece_value = np.sum(bin_totals[nonzero] / len(y_true) * np.abs(
        np.array([y_true[bin_ids == i].mean() for i in np.where(nonzero)[0]]) - np.array([y_probability[bin_ids == i].mean() for i in np.where(nonzero)[0]])
    ))

    return ece_value 

#TSS: True skill statistic - how well the model separates events from things thar are not event from -1 to 1, 0 is random (pretty normal)
#just measuring our separation from when events truly happen to when they actually don't
#HSS: Heidke skill score: measures our accuracy relative to just random chance
#BSS: Brier skill score - how much better are we than just the climatology, negative means worse and whether it's worth it
