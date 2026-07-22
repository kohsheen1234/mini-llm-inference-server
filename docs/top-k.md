# Top-k filtering

> Step 3 of the [roadmap](../README.md#roadmap). Implemented in
> [`mini_llm_server/sampling.py`](../mini_llm_server/sampling.py) as `top_k_filter`.

Top-k filtering restricts sampling to the **k most likely tokens**. Everything
outside the top k is removed from consideration before the final draw, which
cuts off the long, noisy tail of low-probability tokens while still allowing
variety among the plausible ones.

It is applied to the logits, after [temperature](temperature.md) and before
[softmax](stable-softmax.md).

## The idea

Rather than deleting entries (which would change the array's shape), we **mask**
the rejected logits to `-inf`. When those masked logits reach softmax,
$e^{-\infty} = 0$, so they receive exactly zero probability and the surviving k
tokens renormalise among themselves:

```text
logits      [3.0, 2.0, 1.0, 0.1]
top-2 mask  [3.0, 2.0, -inf, -inf]
softmax     [0.731, 0.269, 0.0, 0.0]
```

## How it works

```python
def top_k_filter(logits, k):
    vocab_size = logits.shape[-1]
    if k >= vocab_size:
        return logits
    if k <= 0:
        return np.full_like(logits, -np.inf)

    threshold = np.partition(logits, -k, axis=-1)[..., -k]
    threshold = np.expand_dims(threshold, axis=-1)
    return np.where(logits >= threshold, logits, -np.inf)
```

**Find the threshold.** `np.partition(logits, -k, axis=-1)` rearranges each row
so the element at index `-k` is exactly the k-th largest value, with everything
larger to its right — without paying for a full sort. Indexing `[..., -k]` picks
that k-th largest value out as the per-row cutoff.

**Restore the axis.** `np.partition` reduces the last axis away, so
`np.expand_dims(..., axis=-1)` puts a length-1 axis back. That lets the `(batch,
1)` threshold broadcast against the `(batch, vocab)` logits.

**Mask.** `np.where(logits >= threshold, logits, -np.inf)` keeps values at or
above the cutoff and sends the rest to `-inf`.

## Edge cases

- **`k >= vocab_size`** — nothing to filter, so the logits are returned
  unchanged.
- **`k <= 0`** — no tokens are allowed; every logit becomes `-inf`. (Feeding this
  straight into softmax gives `0/0 = NaN`, so treat `k <= 0` as a degenerate
  configuration, not a normal sampling mode.)
- **Ties at the cutoff** — the comparison is `>=`, so every logit equal to the
  k-th largest value is kept. When the cutoff value is duplicated, more than `k`
  tokens can survive. This keeps the operation deterministic and avoids
  arbitrarily breaking ties.

## Example

```python
import numpy as np
from mini_llm_server import top_k_filter, stable_softmax

logits = np.array([3.0, 2.0, 1.0, 0.1])

filtered = top_k_filter(logits, k=2)
print(filtered)                  # [ 3.  2. -inf -inf]
print(stable_softmax(filtered))  # [0.73106 0.26894 0.  0.]
```

Batched input filters each row independently:

```python
logits = np.array([
    [2.0, 1.0, 0.1],
    [5.0, 4.0, 3.0],
])
print(top_k_filter(logits, k=1))
# [[ 2. -inf -inf]
#  [ 5. -inf -inf]]
```

## Where top-k sits in inference

```text
transformer output → logits → apply_temperature → top_k_filter → stable softmax → probabilities → token sampling
```

Top-k is a hard cutoff on *count*. Top-p (nucleus) filtering — a later roadmap
step — instead cuts on *cumulative probability mass*, keeping the smallest set of
tokens whose probabilities sum past a threshold.
