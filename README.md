# Mini LLM Inference Server

A small Large Language Model inference server built from first principles — not
to train a model, but to understand what happens *after* a model receives a
prompt.

```text
prompt → tokenization → embeddings → transformer inference → logits → sampling → next token → repeat
```

The pipeline is built incrementally as small, self-contained components, each
solving one part of inference and then assembled into the final server.

## Why build this?

High-level libraries make inference look like a single line:

```python
output = model.generate(tokens)
```

That line hides real systems problems this project makes explicit: how logits
become probabilities, how tokens are sampled, how temperature shapes generation,
how the model avoids recomputing attention (KV cache), how that cache memory is
allocated, how requests are batched, and how a server trades off throughput and
latency.

The goal is something small enough to read end-to-end, but complete enough to
show the core mechanics behind production inference systems.

## Project layout

```text
mini-llm-inference-server/
├── mini_llm_server/        # the package
│   ├── __init__.py         # public API re-exports
│   └── sampling.py         # sampling primitives (softmax, temperature, …)
├── docs/                   # per-component deep dives
│   ├── stable-softmax.md
│   ├── temperature.md
│   └── softmax.svg
├── examples/
│   └── plot_softmax.py     # regenerates docs/softmax.svg (stdlib only)
├── requirements.txt
└── README.md
```

As later stages land, new modules join the package (`tokenizer.py`,
`transformer.py`, `kv_cache.py`, `scheduler.py`, `server.py`), each paired with a
doc under `docs/`.

## Quickstart

```bash
python -m pip install -r requirements.txt
```

```python
import numpy as np
from mini_llm_server import stable_softmax, apply_temperature

logits = np.array([2.0, 1.0, 0.1])
probs = stable_softmax(apply_temperature(logits, temperature=0.8))
print(probs)  # non-negative, sums to 1
```

Run from the repository root so `mini_llm_server` is importable.

## Roadmap

Progress is tracked component by component. Completed steps link to their deep
dive.

### Part 1 — Sampling primitives

- [x] [Numerically stable softmax](docs/stable-softmax.md)
- [x] [Temperature scaling](docs/temperature.md)
- [x] [Top-k filtering](docs/top-k.md)
- [x] [Top-p filtering](docs/top-p.md)
- [ ] Sampling from probabilities
- [ ] Greedy token selection

### Part 2 — Tokenization

- [ ] Build a vocabulary
- [ ] Encode prompts
- [ ] Decode tokens

### Part 3 — Tiny Transformer with KV cache

- [ ] Embed tokens
- [ ] Linear projection
- [ ] Initialize KV cache
- [ ] Append key and value states
- [ ] Causal attention
- [ ] Model prefill
- [ ] Decode one token at a time

### Later stages

- [ ] Paged KV-cache management
- [ ] Request batching and scheduling
- [ ] Throughput and latency measurement
- [ ] End-to-end inference server

## Core idea

An LLM inference server is not one large function. It is a pipeline of small
components with strict contracts:

```text
correct shapes + stable numerical operations + controlled memory use + predictable scheduling = reliable inference
```

This project builds those contracts one at a time.

## Attribution

An independent implementation created while working through the Deep-ML mini LLM
inference server project. The code, explanations, and documentation here are
written for learning and portfolio purposes. Platform-owned hidden tests and
restricted instructional content are not included.
