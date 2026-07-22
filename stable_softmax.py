import numpy as np

def stable_softmax(logits):
    # TODO: compute a numerically stable softmax over the last axis of logits.
    maximum = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - maximum
    exponentials = np.exp(shifted)
    total = np.sum(exponentials, axis=-1, keepdims=True)
    probabilities = exponentials / total
    return probabilities
