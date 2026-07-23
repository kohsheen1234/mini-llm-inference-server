# Building a vocabulary

> Step 7 of the [roadmap](../README.md#roadmap), and the first of Part 2
> (Tokenization). Implemented in
> [`mini_llm_server/tokenizer.py`](../mini_llm_server/tokenizer.py) as `build_vocab`.

## Why this step exists

A language model only ever sees integer **ids**, never raw text. Before we can
run prefill or decode, we need a deterministic way to map strings to ids and
back. `build_vocab` builds that mapping once from a corpus, so every later stage
— encoding prompts, decoding outputs, embedding lookups — shares the same id
space.

## The concept

A vocabulary is two parallel structures that invert each other:

- **`id_to_token`** — a list where index `i` holds the token string with id `i`.
- **`token_to_id`** — a dict mapping each token back to its id.

Two ordering rules keep the mapping stable and reproducible:

1. **Special tokens get the lowest ids**, in the exact order given. Reserving
   `<pad>`, `<bos>`, `<eos>`, … up front means their ids never shift when the
   corpus changes — downstream code can safely assume, say, `<pad> == 0`.
2. **Corpus characters follow in sorted order.** Sorting makes the vocabulary
   identical across runs regardless of how the corpus text was ordered.

This implementation is **character-level**: the "tokens" from the corpus are
individual characters.

## How it works

```python
def build_vocab(corpus, special_tokens):
    id_to_token = list(special_tokens)
    special_set = set(special_tokens)

    characters = sorted(set("".join(corpus)))
    id_to_token.extend(c for c in characters if c not in special_set)

    token_to_id = {token: token_id for token_id, token in enumerate(id_to_token)}
    return {"token_to_id": token_to_id, "id_to_token": id_to_token}
```

1. **Seed with the specials.** `id_to_token` starts as a copy of `special_tokens`
   (a copy, so the caller's list is not mutated), preserving their order.
2. **Collect unique characters.** `set("".join(corpus))` flattens every string in
   the corpus into one character set, deduping across strings; `sorted(...)`
   fixes a deterministic order.
3. **Append, skipping collisions.** Each character is added unless a special
   token already claimed it (a special token can literally be a single
   character), checked against `special_set` for an O(1) lookup.
4. **Invert.** `enumerate(id_to_token)` builds `token_to_id` as the exact inverse.

## Common pitfall

Sorting the *whole* vocabulary together — specials and characters mixed — would
place `<bos>` at id 0 only by luck of string ordering, and downstream code that
assumes `<pad> == 0` would break silently. Specials must be seeded **first and in
order**, before the sorted characters are appended.

## Edge cases

- **Empty corpus** → only the special tokens are returned.
- **Duplicate characters across strings** → deduped by the `set`.
- **Whitespace** → treated as an ordinary character and included.
- **A special token that is also a corpus character** → kept once, at its special
  id; not duplicated among the sorted characters.

## Example

```python
from mini_llm_server import build_vocab

vocab = build_vocab(["hi"], ["<pad>", "<bos>"])
print(vocab["id_to_token"])        # ['<pad>', '<bos>', 'h', 'i']
print(vocab["token_to_id"]["h"])   # 2
```

## Where it sits in inference

```text
corpus + special tokens → build_vocab → { token_to_id, id_to_token }
                                              ↓
                          encode_prompt / decode_tokens (next steps)
```

The next steps, `encode_prompt` and `decode_tokens`, use this mapping to turn
prompt strings into id sequences and model outputs back into text.
