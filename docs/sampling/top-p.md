# Top-p (nucleus) filtering

> Step 4 of the [roadmap](../../README.md#roadmap). Implemented in
> [`mini_llm_server/sampling.py`](../../mini_llm_server/sampling.py) as `top_p_filter`.

Top-p filtering — also called **nucleus sampling** — keeps the *smallest* set of
tokens whose probabilities add up to at least `p`, and masks the rest to `-inf`.
Where [top-k](top-k.md) fixes the number of tokens, top-p fixes the amount of
probability *mass* and lets the count adapt to the shape of the distribution:

- A peaked distribution (one token dominates) → the nucleus is small.
- A flat distribution (many plausible tokens) → the nucleus is large.

It is applied to the logits, after [temperature](temperature.md) and before
[softmax](stable-softmax.md).

## The idea

Rank the tokens by probability, walk down the list accumulating mass, and stop
once the running total crosses `p`. The token that crosses the threshold is
**kept** (so the nucleus always reaches `p`), and everything past it is dropped:

```text
logits              [3.0, 2.0, 1.0, 0.1]
probabilities       [0.642, 0.236, 0.087, 0.035]
cumulative          [0.642, 0.878, 0.965, 1.000]

p = 0.9  ->  keep the first 3 (0.965 is the first total ≥ 0.9)
mask                [3.0, 2.0, 1.0, -inf]
softmax             [0.665, 0.245, 0.090, 0.0]
```

With `p = 0.8` the same logits keep only the first two tokens (`0.878 ≥ 0.8`),
which shows the count adapting to `p`.

## How it works

```python
def top_p_filter(logits, p):
    probabilities = stable_softmax(logits)

    sorted_indices = np.argsort(-probabilities, axis=-1, kind="stable")
    sorted_probabilities = np.take_along_axis(probabilities, sorted_indices, axis=-1)

    cumulative = np.cumsum(sorted_probabilities, axis=-1)
    previous_cumulative = np.concatenate(
        [np.zeros_like(cumulative[..., :1]), cumulative[..., :-1]],
        axis=-1,
    )

    sorted_remove_mask = previous_cumulative >= p

    remove_mask = np.zeros_like(sorted_remove_mask, dtype=bool)
    np.put_along_axis(remove_mask, sorted_indices, sorted_remove_mask, axis=-1)
    return np.where(remove_mask, -np.inf, logits)
```

**Sort by probability.** `argsort(-probabilities, kind="stable")` gives the
indices from most to least likely; the stable sort breaks ties by original
position so the result is deterministic. `take_along_axis` reorders the
probabilities to match.

**Accumulate mass.** `np.cumsum` gives the running total in ranked order.
`previous_cumulative` shifts that total one position right (prepending a zero),
so each entry holds the mass accumulated *before* its token.

**Keep the crossing token.** The mask is `previous_cumulative >= p`, not
`cumulative >= p`. Comparing against the mass *before* a token means a token is
removed only once earlier tokens have already reached `p` — so the token that
first crosses the threshold is kept, and the top token is always kept.

**Scatter back.** The mask is in ranked order, so `np.put_along_axis` writes each
flag back to its token's original position before `np.where` applies it.

## Edge cases

- **`p >= 1.0`** — every token is needed to reach the mass, so nothing is masked
  (the full distribution passes through).
- **`p <= 0`** — `previous_cumulative >= p` is true everywhere, including the top
  token, so every logit becomes `-inf`. As with top-k's `k <= 0`, this is a
  degenerate configuration (softmax of all `-inf` is `NaN`), not a normal mode.
- **Exact boundaries and floating point** — because probabilities are summed in
  floating point, a total that is mathematically exactly `p` can land a hair
  above or below it, occasionally shifting the nucleus by one token. This is
  inherent to nucleus sampling and not specific to this implementation.

## Example

```python
import numpy as np
from mini_llm_server import top_p_filter, stable_softmax

logits = np.array([3.0, 2.0, 1.0, 0.1])

filtered = top_p_filter(logits, p=0.9)
print(filtered)                  # [ 3.  2.  1. -inf]
print(stable_softmax(filtered))  # [0.66524 0.24473 0.09003 0.]

print(top_p_filter(logits, p=0.8))  # [ 3.  2. -inf -inf]
```

## Where top-p sits in inference

```text
transformer output → logits → apply_temperature → top_p_filter → stable softmax → probabilities → token sampling
```

Top-k and top-p are usually alternatives: pick one to truncate the tail. They can
also be combined (top-k then top-p) for a hard cap on both count and mass.
