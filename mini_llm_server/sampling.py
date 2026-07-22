"""Sampling primitives: turning raw model logits into a token distribution.

A transformer emits one logit (an unbounded score) per vocabulary token. Before
a token can be sampled, those logits are optionally reshaped by temperature and
then normalised into a probability distribution by softmax:

    logits -> apply_temperature -> top_k_filter -> stable_softmax -> probabilities -> sample

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


def top_k_filter(logits, k):
    """Keep only the ``k`` largest logits per row, masking the rest to ``-inf``.

    Restricts sampling to the ``k`` most likely tokens. Masked positions become
    ``-inf`` so a subsequent ``stable_softmax`` assigns them zero probability.
    ``k >= vocab_size`` is a no-op; ``k <= 0`` masks everything. Ties at the
    k-th largest value are all kept, so more than ``k`` logits can survive when
    there are duplicates. See ``docs/top-k.md``.
    """
    vocab_size = logits.shape[-1]
    if k >= vocab_size:
        return logits
    if k <= 0:
        return np.full_like(logits, -np.inf)

    # The k-th largest value in each row is the keep/mask threshold.
    threshold = np.partition(logits, -k, axis=-1)[..., -k]
    threshold = np.expand_dims(threshold, axis=-1)  # restore axis for broadcasting
    return np.where(logits >= threshold, logits, -np.inf)
