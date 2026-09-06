import numpy as np


def evaluate_measures(sample):
    sample = np.array(sample)
    n = len(sample)

    if n == 0:
        return {"gini": 0.0, "entropy": 0.0, "error": 0.0}

    unique_classes, counts = np.unique(sample, return_counts=True)
    probabilities = counts / n

    gini = 1 - np.sum(probabilities**2)

    entropy = -np.sum([p * np.log(p) if p > 0 else 0 for p in probabilities])

    error = 1 - np.max(probabilities)

    measures = {"gini": float(gini), "entropy": float(entropy), "error": float(error)}

    return measures
