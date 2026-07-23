# Greedy token selection

> Step 6 of the [roadmap](../../README.md#roadmap). Implemented in
> [`mini_llm_server/sampling.py`](../../mini_llm_server/sampling.py) as `greedy_select`.

Greedy selection is the **deterministic** way to pick the next token: instead of
drawing from the distribution like [`sample_from_probs`](sample-from-probs.md),
it always chooses the single highest-scoring token.

## The idea

```python
def greedy_select(logits):
    return int(np.argmax(logits))
```

`np.argmax` returns the index of the largest value; wrapping it in `int(...)`
gives a plain Python token id.

## Runs directly on logits

Greedy decoding does **not** need a softmax step. Softmax is *order-preserving*
(it never changes which token is largest), so:

```text
argmax(logits) == argmax(softmax(logits))
```

You can therefore call `greedy_select` on raw logits and skip the normalisation
entirely — it is both simpler and cheaper than sampling. (It still works fine if
you happen to pass probabilities.)

## Behaviour and expectations

- **Deterministic.** No randomness, no generator — the same logits always give
  the same token. Useful for reproducible output and for decoding where you want
  the "most likely" continuation.
- **Ties resolve to the lowest index.** When two logits are equal and maximal,
  `np.argmax` returns the first one, so the result stays deterministic.
- **1-D input.** Operates on a single logit vector (one token per call), matching
  `sample_from_probs`.

## Greedy vs. sampling

| | `greedy_select` | `sample_from_probs` |
|---|---|---|
| Output | always the argmax | a random draw weighted by probability |
| Randomness | none | needs an `rng` |
| Needs softmax | no (works on logits) | yes (needs a normalised distribution) |
| Diversity | none — same output every time | varied output run to run |

Greedy is equivalent to sampling at temperature 0 — which is exactly why
[`apply_temperature`](temperature.md) treats `temperature <= 0` as greedy and
leaves the logits untouched for this step to act on.

## Example

```python
import numpy as np
from mini_llm_server import greedy_select

logits = np.array([3.0, 2.0, 1.0, 0.1])
print(greedy_select(logits))            # 0 — the highest logit

print(greedy_select(np.array([1.0, 5.0, 5.0, 2.0])))  # 1 — ties pick lowest index
```

## Where it sits in inference

```text
transformer output → logits → greedy_select → token id
```

Greedy is a complete decoding strategy on its own — the temperature/top-k/top-p/
softmax path is only needed when you want *randomised* sampling.
