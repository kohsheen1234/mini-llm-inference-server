"""Mini LLM inference server — a from-scratch inference pipeline, built incrementally.

Public API is re-exported here so callers can write::

    from mini_llm_server import (
        stable_softmax, apply_temperature, top_k_filter, top_p_filter,
        sample_from_probs, greedy_select,
    )
"""

from mini_llm_server.sampling import (
    apply_temperature,
    greedy_select,
    sample_from_probs,
    stable_softmax,
    top_k_filter,
    top_p_filter,
)

__all__ = [
    "apply_temperature",
    "greedy_select",
    "sample_from_probs",
    "stable_softmax",
    "top_k_filter",
    "top_p_filter",
]
