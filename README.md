# Mini LLM Inference Server

This project builds a small Large Language Model inference server from first principles.

The goal is not to train a large model. The goal is to understand what happens after a model receives a prompt:

```text
prompt
  ↓
tokenization
  ↓
embeddings
  ↓
transformer inference
  ↓
logits
  ↓
sampling
  ↓
next token
  ↓
repeat
```

The implementation is developed incrementally through a sequence of small, testable components. Each component solves one part of the inference pipeline and is then assembled into the final server.

## Project status

Current progress:

```text
1 / 51 steps complete
```

Implemented:

- Numerically stable softmax

Planned visible modules:

- Sampling primitives
- Tokenization
- Tiny Transformer inference
- KV caching
- Paged KV caching
- Request scheduling
- Batched generation
- End-to-end inference serving

## Why build this?

High-level libraries make LLM inference look simple:

```python
output = model.generate(tokens)
```

But that one line hides many systems problems:

- How are raw model scores converted into probabilities?
- How are tokens sampled?
- How does temperature affect generation?
- How does the model avoid recomputing old attention states?
- How is KV-cache memory allocated?
- How are multiple requests batched?
- How does a server balance throughput and latency?

This project exposes those hidden steps.

The final result should be small enough to understand, but complete enough to show the core mechanics behind production LLM inference systems.

## Step 1: Numerically stable softmax

The first component converts model outputs called **logits** into probabilities.

### What are logits?

A model produces one score for every token in the vocabulary.

For example:

```python
logits = np.array([2.0, 1.0, 0.1])
```

These values are not probabilities.

They:

- can be negative
- can be larger than 1
- do not need to sum to 1

Softmax converts them into a probability distribution.

```text
logits       → softmax → probabilities
[2, 1, 0.1]             [0.659, 0.242, 0.099]
```

The output values are non-negative and sum to 1.

![Two panels. Left: e^(logit) plotted on a log scale, rising in a straight line until it crosses the float64 maximum near a logit of 710 and overflows to inf, then NaN. Right: a bar chart showing that softmax of the raw logits [2, 1, 0.1] and of the max-shifted logits [0, -1, -1.9] produce identical probabilities.](docs/softmax.svg)

*Left: naive `exp()` overflows for large logits. Right: subtracting the max leaves the probabilities unchanged. The rest of this section derives both facts.*

### Basic softmax formula

Softmax turns a vector of logits $x = (x_1, x_2, \dots, x_n)$ into a probability
distribution. For each entry $x_i$:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

Three properties follow directly from this definition, and together they are
exactly what "a probability distribution" means:

- **Non-negative.** $e^{x_i} > 0$ for every real $x_i$, so every output is positive.
- **Sums to one.** The denominator is the sum of the numerators, so
  $\sum_{i=1}^{n} \text{softmax}(x)_i = 1$.
- **Order-preserving.** $e^{x}$ is strictly increasing, so a larger logit always
  maps to a larger probability. Softmax reshapes the scores without reordering them.

A direct implementation would be:

```python
exp_values = np.exp(logits)
probabilities = exp_values / np.sum(exp_values)
```

This works for small values, but it can fail for large logits.

### Why the naive version is unsafe

Floating-point numbers have a finite range. The largest finite value a 64-bit
float can hold is about $1.8 \times 10^{308}$, and

$$e^{x} > 1.8 \times 10^{308} \quad\text{once}\quad x > \ln(1.8 \times 10^{308}) \approx 709.8$$

So any logit above roughly **710** overflows `np.exp` to `inf` (panel A above).
Concretely, consider:

```python
logits = np.array([1000.0, 999.0, 998.0])
```

Computing this directly:

```python
np.exp(1000.0)
```

produces a number too large for normal floating-point representation.

The result may become:

```text
inf
```

The later division `inf / inf` then produces:

```text
NaN
```

That `NaN` propagates into token sampling and breaks generation. Note the logits
themselves are perfectly ordinary — it is only the intermediate `exp` that
overflows.

### The stability trick

Softmax is **invariant to shifting all logits by the same constant**. For any
constant $c$:

$$\text{softmax}(x - c)_i = \frac{e^{x_i - c}}{\sum_{j} e^{x_j - c}}
= \frac{e^{x_i}\,e^{-c}}{\sum_{j} e^{x_j}\,e^{-c}}
= \frac{e^{-c}}{e^{-c}} \cdot \frac{e^{x_i}}{\sum_{j} e^{x_j}}
= \text{softmax}(x)_i$$

The shared factor $e^{-c}$ cancels top and bottom, so the output is identical
(panel B above). We are free to pick any $c$ — so pick the one that makes the
exponentials safe:

$$c = \max_{j} x_j$$

With this choice every shifted logit satisfies $x_i - c \le 0$, so

$$e^{x_i - c} \in (0, 1]$$

- **No overflow:** the largest exponent is $e^{0} = 1$, well within range.
- **No divide-by-zero:** at least one term equals $1$, so the denominator is $\ge 1$.

For:

```text
[1000, 999, 998]
```

subtracting the maximum gives:

```text
[0, -1, -2]
```

Now exponentiation is safe:

```text
[e^0, e^-1, e^-2]
=
[1.0, 0.3679, 0.1353]
```

The relative differences — and therefore the final probabilities — are unchanged.

### The log-sum-exp view

The same trick has a name outside softmax. The denominator's logarithm is the
**log-sum-exp** (LSE) function:

$$\text{LSE}(x) = \log \sum_{j} e^{x_j}$$

and softmax in log-space is simply $\log \text{softmax}(x)_i = x_i - \text{LSE}(x)$.
Computed naively, `LSE` overflows for the same reason softmax does. The stable
form factors the max back out:

$$\text{LSE}(x) = \max_{j} x_j + \log \sum_{j} e^{x_j - \max_{k} x_k}$$

This is the identical "subtract the max, then exponentiate" pattern, and it is
why the implementation below shifts by `np.max` before calling `np.exp`.

## Implementation

```python
import numpy as np


def stable_softmax(logits):
    maximum = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - maximum
    exponentials = np.exp(shifted)
    total = np.sum(exponentials, axis=-1, keepdims=True)
    probabilities = exponentials / total
    return probabilities
```

## Line-by-line explanation

### 1. Import NumPy

```python
import numpy as np
```

- `numpy` is a Python library for arrays and numerical computation.
- `as np` creates the shorter name `np`.
- `np.array`, `np.max`, `np.exp`, and `np.sum` are NumPy operations.

### 2. Define the function

```python
def stable_softmax(logits):
```

- `def` creates a function.
- `stable_softmax` is the function name.
- `logits` is the input array.
- The colon starts the function body.

### 3. Find the largest value

```python
maximum = np.max(logits, axis=-1, keepdims=True)
```

`np.max` finds the largest value.

#### `axis=-1`

`axis=-1` means:

> Apply the operation along the final dimension.

This matters because logits may have several shapes.

One request:

```text
(vocabulary_size,)
```

Several requests:

```text
(batch_size, vocabulary_size)
```

A sequence batch:

```text
(batch_size, sequence_length, vocabulary_size)
```

In every case, the final axis represents vocabulary scores.

Example:

```python
logits = np.array([
    [2.0, 1.0, 0.0],
    [4.0, 3.0, 2.0],
])
```

Shape:

```text
(2, 3)
```

Using:

```python
np.max(logits, axis=-1)
```

returns:

```text
[2.0, 4.0]
```

One maximum is computed for each row.

#### `keepdims=True`

Without `keepdims=True`, the maximum has shape:

```text
(2,)
```

With `keepdims=True`, it has shape:

```text
(2, 1)
```

That preserved dimension makes subtraction work cleanly:

```python
shifted = logits - maximum
```

NumPy broadcasts each row maximum across its row.

### 4. Shift the logits

```python
shifted = logits - maximum
```

The largest value in each group becomes zero.

Every other value becomes zero or negative.

That prevents exponentiation from overflowing.

### 5. Exponentiate

```python
exponentials = np.exp(shifted)
```

`np.exp(x)` computes:

$$e^x$$

All outputs are positive.

### 6. Compute the denominator

```python
total = np.sum(exponentials, axis=-1, keepdims=True)
```

This adds the exponential values along the vocabulary dimension.

Again:

- `axis=-1` means the final axis
- `keepdims=True` preserves the dimension for broadcasting

### 7. Normalize

```python
probabilities = exponentials / total
```

Each exponential value is divided by the total.

The values now sum to 1 along the last axis.

### 8. Return the result

```python
return probabilities
```

`return` sends the computed array back to the caller.

## Example

```python
import numpy as np

logits = np.array([2.0, 1.0, 0.1])
probs = stable_softmax(logits)

print(probs)
print(np.sum(probs))
```

Expected output:

```text
[0.65900114 0.24243297 0.09856589]
1.0
```

## Batched example

```python
logits = np.array([
    [2.0, 1.0, 0.0],
    [1000.0, 999.0, 998.0],
])

probs = stable_softmax(logits)

print(probs)
print(np.sum(probs, axis=-1))
```

Expected row sums:

```text
[1.0, 1.0]
```

Each row is treated as a separate probability distribution.

## When softmax is used in LLM inference

During generation, the model produces one logit for every vocabulary token.

Example:

```text
"cat"       → 8.2
"dog"       → 6.4
"airplane"  → 1.1
```

Softmax converts those scores into probabilities:

```text
"cat"       → 0.82
"dog"       → 0.16
"airplane"  → 0.02
```

The sampling system can then:

- choose the highest-probability token
- sample randomly from the distribution
- apply temperature
- apply top-k filtering
- apply top-p filtering

Softmax sits between model computation and token selection.

```text
transformer output
       ↓
     logits
       ↓
temperature and filtering
       ↓
stable softmax
       ↓
probabilities
       ↓
token sampling
```

## Correctness requirements

The implementation must satisfy these properties:

### Same shape

Input:

```text
(..., vocabulary_size)
```

Output:

```text
(..., vocabulary_size)
```

### Non-negative values

Every probability must be at least zero.

### Sum to one

For every distribution:

```python
np.sum(probabilities, axis=-1)
```

should be approximately:

```text
1.0
```

### Numerical stability

Very large positive or negative logits must not produce:

```text
inf
NaN
```

## Tests covered

The current implementation passes tests for:

- Basic one-dimensional input
- Very large logits
- Batched two-dimensional input
- Three-dimensional input
- Probability sums along the final axis

## Repository structure

A suggested repository layout:

```text
mini-llm-inference-server/
├── README.md
├── inference_server.py
├── requirements.txt
├── tests/
│   └── test_inference_server.py
└── examples/
    └── stable_softmax_example.py
```

As the project grows, the implementation can be split into modules:

```text
mini-llm-inference-server/
├── README.md
├── requirements.txt
├── src/
│   ├── sampling.py
│   ├── tokenizer.py
│   ├── transformer.py
│   ├── kv_cache.py
│   ├── scheduler.py
│   └── server.py
├── tests/
└── examples/
```

## Running locally

Install NumPy:

```bash
python -m pip install numpy
```

Run the implementation:

```bash
python inference_server.py
```

Run tests with pytest:

```bash
python -m pip install pytest
pytest
```

## Development roadmap

### Part 1: Sampling primitives

- [x] Stable softmax
- [ ] Temperature scaling
- [ ] Top-k filtering
- [ ] Top-p filtering
- [ ] Sampling from probabilities
- [ ] Greedy token selection

### Part 2: Tokenization

- [ ] Build a vocabulary
- [ ] Encode prompts
- [ ] Decode tokens

### Part 3: Tiny Transformer with KV cache

- [ ] Embed tokens
- [ ] Linear projection
- [ ] Initialize KV cache
- [ ] Append key and value states
- [ ] Causal attention
- [ ] Model prefill
- [ ] Decode one token at a time

### Later stages

- [ ] Paged KV-cache management
- [ ] Request batching
- [ ] Scheduling
- [ ] Throughput and latency measurement
- [ ] End-to-end inference server

## Core idea

An LLM inference server is not one large function.

It is a pipeline of small components with strict contracts:

```text
correct shapes
+ stable numerical operations
+ controlled memory use
+ predictable scheduling
= reliable inference
```

This project builds those contracts one at a time.

## Attribution

This repository is an independent implementation created while working through the Deep-ML mini LLM inference server project.

The code, explanations, examples, and documentation in this repository are written for learning and portfolio purposes. Platform-owned hidden tests and restricted instructional content are not included.
