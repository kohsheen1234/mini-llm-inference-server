"""Tokenization: mapping between text and the integer ids a model consumes.

A model only ever sees integer ids, never raw text. This module builds the
vocabulary that fixes that mapping once, so every later stage — encoding
prompts, decoding outputs, embedding lookups — shares a single id space.
"""


def build_vocab(corpus, special_tokens):
    """Build a character-level vocabulary from a corpus.

    ``special_tokens`` occupy the lowest ids, in the exact order given, so ids
    like ``<pad> == 0`` stay stable regardless of corpus content. The remaining
    unique characters from ``corpus`` follow in sorted order for reproducibility;
    a character already claimed by a special token is not added twice. Returns a
    dict with ``token_to_id`` (dict) and ``id_to_token`` (list), which invert
    each other. See ``docs/build-vocab.md``.
    """
    # Special tokens take the first ids, in caller order.
    id_to_token = list(special_tokens)
    special_set = set(special_tokens)

    # Then every unique character in the corpus, in deterministic sorted order,
    # skipping any a special token already claimed.
    characters = sorted(set("".join(corpus)))
    id_to_token.extend(c for c in characters if c not in special_set)

    token_to_id = {token: token_id for token_id, token in enumerate(id_to_token)}
    return {"token_to_id": token_to_id, "id_to_token": id_to_token}
