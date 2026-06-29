import collections

def make_splits(records, val_frac = 0.15, test_frac = 0.15):

    ar_first_seen = collections.defaultdict(lambda: "9999")

    for r in records:

        if r["timestamp"] < ar_first_seen[r["ar_id"]]:

            ar_first_seen[r["ar_id"]] = r["timestamp"]

    sorted_arrays = sorted(ar_first_seen, key = lambda a: ar_first_seen[a])
    length = len(sorted_arrays)

    max_test = max(1, int(length * test_frac))
    max_val = max(1, int(length * val_frac))

    n_train = length - max_val - max_test

    temp = {}

    for array in sorted_arrays[: n_train]:

        temp[array] = "train"
    
    for array in sorted_arrays[n_train: n_train + max_val]:

        temp[array] = "val"
    
    for array in sorted_arrays[n_train + max_val:]:
        temp[array] = "test"

    
    return temp

def apply_splits(records, temp):

    train = []
    validation = []
    test = []

    for r in records:
        bucket = temp.get(r["ar_id"])
        if bucket == "train":
            train.append(r)
        elif bucket == "val":
            validation.append(r)
        elif bucket == "test":
            test.append(r)
        
    return train, validation, test

def no_overlap(train, validation, test):

    train_arrays = {r["ar_id"] for r in train}
    validation_arrays = {r["ar_id"] for r in validation}
    test_arrays = {r["ar_id"] for r in test}

    assert train_arrays.isdisjoint(validation_arrays), f"Overlap train/val: {train_arrays & validation_arrays}"
    assert train_arrays.isdisjoint(test_arrays), f"Overlap train/test: {train_arrays & test_arrays}"
    assert test_arrays.isdisjoint(validation_arrays), f"Overlap test/val: {test_arrays & validation_arrays}"


