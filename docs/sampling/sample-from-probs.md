# Sampling from probabilities

> Step 5 of the [roadmap](../../README.md#roadmap). Implemented in
> [`mini_llm_server/sampling.py`](../../mini_llm_server/sampling.py) as `sample_from_probs`.

This is the final step of the sampling pipeline: given a probability
distribution over the vocabulary, actually **draw one token**. Everything before
it — [temperature](temperature.md), [top-k](top-k.md), [top-p](top-p.md),
[softmax](stable-softmax.md) — only *shapes* the distribution; this is where a
concrete token id is chosen.

## The idea

`probs` is a categorical distribution: one non-negative probability per token,
summing to 1. Drawing from it means picking token `i` with probability
`probs[i]`. NumPy's generator does exactly this:

```python
def sample_from_probs(probs, rng):
    return int(rng.choice(len(probs), p=probs))
```

`rng.choice(n, p=probs)` selects an index in `0..n-1` according to the given
probabilities; wrapping it in `int(...)` returns a plain Python token id.

## Why pass in the generator?

The function takes an explicit `rng` (a `numpy.random.Generator`) rather than
calling the global `np.random`. This is a deliberate, production-minded choice:

- **Reproducibility.** Seeding one generator (`np.random.default_rng(seed)`) makes
  an entire run repeatable, which matters for tests and debugging.
- **Isolation.** Separate requests or threads can each own a generator without
  fighting over shared global state.
- **Explicitness.** Randomness is a visible input, not a hidden side effect.

## Guarantees and expectations

- **Expects a normalised distribution.** `probs` must sum to 1 (within floating
  tolerance); `rng.choice` raises `ValueError` otherwise. The output of
  `stable_softmax` satisfies this by construction.
- **Never draws a masked token.** After top-k / top-p, rejected tokens have
  probability exactly 0, so they can never be selected.
- **Deterministic given a seed.** The same seed and same `probs` yield the same
  draw.
- **1-D input.** This step operates on a single distribution (one token per
  call), unlike the batched filters above it.

## Example

```python
import numpy as np
from mini_llm_server import stable_softmax, sample_from_probs

rng = np.random.default_rng(42)
probs = stable_softmax(np.array([3.0, 2.0, 1.0, 0.1]))
# probs ≈ [0.642, 0.236, 0.087, 0.035]

token_id = sample_from_probs(probs, rng)
print(token_id)  # e.g. 0 — token 0 is drawn ~64% of the time
```

Over many draws the empirical frequencies converge to `probs`:

```python
rng = np.random.default_rng(42)
draws = [sample_from_probs(probs, rng) for _ in range(20000)]
print(np.bincount(draws) / 20000)  # ≈ [0.640 0.238 0.087 0.035]
```

## Where it sits in inference

```text
transformer output → logits → apply_temperature → top-k / top-p → stable softmax → sample_from_probs → token id
```

The returned token id is fed back into the model as the next input, and the loop
repeats until an end-of-sequence token or a length limit. Greedy decoding — the
next roadmap step — replaces this random draw with a deterministic `argmax`.
