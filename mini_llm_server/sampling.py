"""Sampling primitives: turning raw model logits into a token distribution.

A transformer emits one logit (an unbounded score) per vocabulary token. Before
a token can be sampled, those logits are optionally reshaped by temperature and
then normalised into a probability distribution by softmax:

    logits -> apply_temperature -> stable_softmax -> probabilities -> sample

Each function operates over the last axis, so it works for a single request
``(vocab,)``, a batch ``(batch, vocab)``, or a sequence batch
``(batch, seq, vocab)`` without changes.
"""

import numpy as np


def stable_softmax(logits):
    """Convert logits into a probability distribution over the last axis.

    Subtracts the per-row maximum before exponentiating, so arbitrarily large
    logits cannot overflow ``np.exp`` to ``inf``. The output is non-negative and
    sums to 1 along the last axis. See ``docs/stable-softmax.md`` for the
    derivation.
    """
    maximum = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - maximum
    exponentials = np.exp(shifted)
    total = np.sum(exponentials, axis=-1, keepdims=True)
    return exponentials / total


def apply_temperature(logits, temperature):
    """Scale logits by ``1 / temperature`` to sharpen or flatten the distribution.

    ``temperature < 1`` sharpens (more confident, more deterministic),
    ``temperature > 1`` flattens (more diverse). A temperature of 0 or less means
    greedy decoding, so the logits are returned unchanged. See
    ``docs/temperature.md`` for details.
    """
    if temperature <= 0:
        return logits
    return logits / temperature
