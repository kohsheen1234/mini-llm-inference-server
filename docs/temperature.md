# Temperature scaling

> Step 2 of the [roadmap](../README.md#roadmap). Implemented in
> [`mini_llm_server/sampling.py`](../mini_llm_server/sampling.py) as `apply_temperature`.

Temperature is the knob that controls how *adventurous* generation is. It is
applied to the logits **before** [softmax](stable-softmax.md), reshaping the
distribution without changing which token is most likely.

## The formula

$$\text{scaled}(x)_i = \frac{x_i}{T}$$

Dividing every logit by a temperature $T$ and then taking softmax is equivalent
to raising each probability to the power $1/T$ and renormalising:

$$\text{softmax}\!\left(\tfrac{x}{T}\right)_i = \frac{p_i^{1/T}}{\sum_j p_j^{1/T}}
\quad\text{where}\quad p = \text{softmax}(x)$$

That exponent is what sharpens or flattens the distribution.

## What each range does

| Temperature | Effect on logits | Effect on sampling |
|-------------|------------------|--------------------|
| `T -> 0`    | differences blow up | approaches greedy (always the top token) |
| `T < 1`     | differences grow    | **sharper** — more confident, less random |
| `T = 1`     | logits unchanged    | the model's own distribution |
| `T > 1`     | differences shrink  | **flatter** — more diverse, more random |
| `T -> inf`  | all logits equal    | approaches a uniform distribution |

## The greedy edge case

Mathematically the scaling is undefined at $T = 0$ (division by zero) and
meaningless for $T < 0$ (it would invert the ordering). Since a temperature of
zero is conventionally understood as "greedy decoding" — always pick the
highest-scoring token — the implementation treats **any `T <= 0`** as greedy and
returns the logits unchanged, leaving the actual token choice to a downstream
`argmax`:

```python
def apply_temperature(logits, temperature):
    if temperature <= 0:
        return logits
    return logits / temperature
```

This keeps the function total (never raises, never returns `inf`/`NaN`) and
composes cleanly with softmax and greedy selection.

## Example

```python
import numpy as np
from mini_llm_server import apply_temperature, stable_softmax

logits = np.array([2.0, 1.0, 0.1])

# Sharper: probability mass concentrates on the top token.
print(stable_softmax(apply_temperature(logits, 0.5)))
# [0.86378 0.1169  0.01932]

# Neutral: the model's own distribution.
print(stable_softmax(apply_temperature(logits, 1.0)))
# [0.659   0.24243 0.09857]

# Flatter: probabilities move toward uniform.
print(stable_softmax(apply_temperature(logits, 2.0)))
# [0.50169 0.30429 0.19402]

# Greedy: logits unchanged, choose argmax downstream.
print(apply_temperature(logits, 0.0))
# [2.  1.  0.1]
```

## Where temperature sits in inference

```text
transformer output → logits → apply_temperature → stable softmax → probabilities → token sampling
```

Temperature is the first of the sampling controls; top-k and top-p filtering
(later roadmap steps) narrow the distribution further before the final draw.
