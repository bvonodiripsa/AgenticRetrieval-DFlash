# Embedding pooling fix (2026-07-30)

## The bug

`gi_builder.py`'s in-process embedder (`Qwen/Qwen3-Embedding-0.6B`, used for
every food doc, entity, triple, and query vector in the local index) pooled
embeddings by **averaging every token's hidden state** (mean pooling).

Qwen3-Embedding is a decoder-only model, contrastively fine-tuned to be read
off the **last token's** hidden state (with an instruction prefix on the query
side), not off a mean-pooled average. Causal attention means an early token in
a sequence never sees the tokens after it; only the last token has seen the
whole input. Averaging that back in with all the "hasn't seen full context
yet" tokens dilutes the one representation the model was actually trained to
be good at, which turns out to matter a lot for this corpus.

## How it was found

Investigating why DFlash's answer to "high-calorie protein snack for
long-distance running that fits in a running belt" was missing a product
(`22219407`) that the `AgenticRetrieval` baseline surfaced cleanly:

1. Confirmed the product exists in the local index (`food.payload.parquet`).
2. Confirmed it's absent from every candidate DFlash's retrieval pipeline
   collects (seed entities, graph traversal, vector-augmented food search,
   BM25 keyword expansion) — not an LLM-selection problem, a retrieval-recall
   problem.
3. Bumping the retrieval budget (`seed_entities_k` 10→20, `max_triples`
   40→80, `max_source_chunks` 15→25, `vector_augment_k` 12→20) did not help,
   and in one run made answer quality *worse* (an unrelated hair-gel product
   got included with no self-correction from the LLM this time). This ruled
   out "not enough headroom" as the cause.
4. Directly ranked the target product against the whole 58,233-doc food
   corpus by raw cosine similarity to the query: **rank #800**, score 0.42.
   The top 10 by score (0.67-0.69) were things like "chicken donuts",
   "instant noodles with chicken flavor", and "scallop, girolle and paris
   mushroom bites" — nothing related to the query. A narrow, uniformly-high
   similarity band across semantically unrelated documents is the textbook
   symptom of anisotropic/degenerate sentence embeddings, which mean-pooling
   a causal LM is known to produce.
5. Side-by-side test, same model weights, one query and five docs (one
   relevant, four not): mean pooling put the irrelevant docs within
   **0.08-0.15** of the relevant doc's score; last-token pooling + the
   documented query instruction prefix widened that gap to **0.18-0.23**.
6. Re-embedded the 58,233-doc food corpus with last-token pooling: the target
   product's rank went from **#800 (0.42) to #2 (0.55)**, and the new top-15
   were all genuinely relevant protein/energy snacks — no more chicken donuts
   or mislabeled personal-care items.
7. After re-embedding the full local index and restarting the app, product
   `22219407` appeared as **item #1 and the "Top Pick"** of the live answer.
   Re-ran all 10 predefined questions in `data/food.json` against the fixed
   app end to end: all succeeded, answers stayed on-topic and grounded in
   real products, and wall-clock time (3.5-6.2s) was unchanged from before
   the fix.

## The fix

`gi_builder.py`:
- Tokenizer now loads with `padding_side="left"` — this is what makes
  "last token = index `-1`" true for every row in a batch regardless of that
  row's own length.
- `embed_sync`/`embed_batch_sync`/`EmbedClient.embed`/`EmbedClient.embed_batch`
  now take `is_query: bool = False` and pool via `last_hidden_state[:, -1]`
  instead of a mean over `last_hidden_state`.
- `is_query=True` adds Qwen3-Embedding's recommended instruction prefix
  (`"Instruct: Given a search query, retrieve relevant food product passages
  that answer the query\nQuery:{text}"`) — set only on the question side.
  Document/description text (food docs, entity descriptions, triple
  subject-predicate-object strings) is embedded plain, matching how the
  corpus itself was re-embedded.
- Call sites that embed a real user question (`gi_query.py`, `api.py`,
  `gi_builder.py`'s question-driven build mode, and the benchmark/sweep
  scripts) now pass `is_query=True`. Call sites that embed corpus text for
  clustering/storage (`resolve_entities`, `store_entities`, `store_triples`)
  are left at the default `is_query=False`.

The whole local snapshot (food, entities, triples — 1,831,471 vectors total)
was re-embedded with the corrected pooling and swapped into
`data/local_index/*.vecs.npy` (originals backed up alongside with a
`.pre_pooling_fix` suffix). Re-embedding text per container:

| Container | Text embedded |
|---|---|
| `food` | `embedding_text_fields` from `my.yaml` (`product_title`, `product_title_translated`, `brand`, `claims`, `claims_translated`, `ingredients`, `ingredients_translated`, `allergens`, `allergens_translated`, `pack_size`, `pack_size_translated`, `country_code`), one `field: value` line each |
| `entities` | the existing `description` field (already `"{name}. Relations: {...}"`, matching `gi_builder.py::store_entities`) |
| `triples` | `f"{subject} {predicate} {object}"`, matching `gi_builder.py::store_triples` |

Re-embedding throughput on the H100 box (GPU 1, shared with vLLM + both web
apps on GPU 0): ~1,575-1,645 docs/s. Total corpus took ~19 minutes
(entities: 110s, triples: ~17min, food: ~2min, run separately).

## What's still stale

**Cosmos DB itself was not re-embedded.** `scripts/build_local_index.py`
exports whatever is already in Cosmos's `/e` field; it does not compute
embeddings. The corrected vectors only live in the local snapshot
(`data/local_index/*.vecs.npy`). If the snapshot is ever rebuilt from Cosmos
without also fixing the source data, this bug comes back silently. Two ways
to close that gap, not yet done:

1. Re-embed and re-upload `food`/`entities`/`triples` documents' `/e` field
   in Cosmos with the corrected pooling (a real, ~1.83M-document write job —
   same serial-upload bottleneck flagged in `PROGRESS.md`'s "8-9 hour build"
   section would apply).
2. Or, cheaper: have `scripts/build_local_index.py` re-embed from each
   document's text fields at export time instead of trusting Cosmos's stored
   `/e`, so the local snapshot is always self-consistent regardless of what's
   in Cosmos. This also sidesteps needing `text_fields`/description
   reconstruction logic to live in two places.

Whichever path, don't re-run `scripts/build_local_index.py` against the
current Cosmos data and expect the fix to survive — it will pull the old,
mean-pooled vectors right back in.
