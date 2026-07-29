# GB10 optimization results — local GPU index

Measured 2026-07-27 on the same machine, question and config as
[`baseline_gb10.md`](baseline_gb10.md). Harness: `benchmark_compare.py`, 3 runs
per configuration.

## Headline

| Configuration | Retrieval | vs Cosmos |
|---|---|---|
| Cosmos DB (Sweden Central) | 30.56s | 1x |
| Local GPU index, forward edges only | **0.024s** | **1,270x** |
| Local GPU index, forward + reverse edges | **0.025s** | **1,220x** |

Embedding separately went from **202ms on CPU to 30ms on GPU**.

## Stage detail

| Stage | Cosmos | Local (fwd) | Local (fwd+rev) |
|---|---|---|---|
| embed | 0.030s | 0.030s | 0.030s |
| entity_search | 6.677s | 0.004s | 0.003s |
| graph_traversal | 11.769s | 0.018s | 0.019s |
| source_fetch | 12.113s | 0.002s | 0.002s |
| **total** | **30.559s** | **0.024s** | **0.025s** |

Cosmos timings vary considerably run to run over a cross-continental link —
25.6s in the standalone baseline, 30.6s and 38.3s in two comparison runs. The
local numbers are stable to within a millisecond.

**Read the GB10 speedup with care.** Most of it is network distance to Sweden
Central, which the co-located H100 box does not pay. Against the H100's
published 2.94s retrieval, the same local index predicts roughly 0.1s including
embedding, taking end-to-end from 6.40s to about 4.6s. That is the number worth
quoting.

## Correctness

Local search is exact, so any disagreement with Cosmos is either a snapshot bug
or Cosmos' approximation. It is the latter:

```
seed entities  Jaccard 0.538
entity quality exact search wins — 3 hits Cosmos missed all score >= 0.6123,
               vs Cosmos-only max 0.6118
```

Every result the local index found and Cosmos missed scores strictly higher
than every result Cosmos found and the local index missed. Shared hits agree to
four decimal places (local rank 0 similarity 0.6416 vs Cosmos 0.6415), which
confirms the snapshot is faithful. The gap is DiskANN's product quantization to
192 bytes per vector losing true neighbours.

## The traversal bug is fixed

The baseline reported `0 PK + 30 vec` on every run — graph traversal
contributed nothing. With reverse adjacency:

| Seed entity | Forward only | Forward + reverse |
|---|---|---|
| `royal dansk oatmeal cookies with cranberries (...)` | 30 | 30 |
| `polysorbate 20` | 0 | **192** |
| `sweetened dried cranberry (cranberries, sugar)` | 0 | **1** |

End to end this takes the pipeline from `0 PK + 30 vec -> 30 triples` to
`12 PK + 30 vec -> 40 triples`. Traversal costs 0.02–1.0 ms either way.

## Snapshot

Built by `scripts/build_local_index.py` in 9.7 minutes, pulling 1.83M documents
in parallel across the 20 physical partitions at ~3,600 docs/s.

| Artifact | Size |
|---|---|
| `triples.vecs.npy` | 3.26 GB |
| `entities.vecs.npy` | 0.37 GB |
| `food.vecs.npy` | 0.12 GB |
| `triples.csr.npz` | 17.6 MB |
| payload parquet (3 files) | 72.8 MB |
| **GPU resident** | **3.75 GB** |

Loads in 1.4s. CSR covers 299,897 distinct nodes with 1.59M forward and 1.59M
reverse edges.

## BM25 replacement for full-text

`gi_index.fulltext_food` replaces the Cosmos `FullTextContains` fan-out over the
58,233 food documents: 3 keywords returned 30 hits in 40ms, about 13ms per
keyword.

## What changed

| File | Purpose |
|---|---|
| `retrieval.py` | One implementation of the 5-stage pipeline, two backends |
| `gi_index.py` | GPU vector search, CSR traversal, BM25 |
| `scripts/build_local_index.py` | Cosmos to local snapshot exporter |
| `benchmark_compare.py` | Side-by-side harness with a correctness check |
| `gi_query.py` | Uses `retrieve()`; ~170 lines of duplicated queries removed |
| `api.py` | `_dflash_answer` uses `retrieve()` |
| `gi_builder.py` | Embedder moved to GPU |
| `my.yaml` | New `index:` section selecting the backend |

## Not done

`_stream_dflash_sse` and `_stream_gi_sse` in `api.py` still hold their own
copies of the retrieval logic. They interleave SSE progress events between
stages, so converting them changes the granularity of the progress messages the
web UI shows. Worth doing, but it needs the UI exercised against a running LLM
to verify.

**Resolved 2026-07-28**: `_stream_dflash_sse` now uses `retrieve()` and real
token streaming; `_stream_gi_sse` was dead code (no route referenced it) and
was deleted rather than converted. See `PROGRESS.md` and
`results/h100_comparison.md`.

## Update 2026-07-29: the traversal fix above was incomplete

The `12 PK + 30 vec` numbers in "The traversal bug is fixed" above came from
a **second, bigger bug** in `retrieval.py` that wasn't found until later:
the multi-hop frontier only ever advanced past hop 0 by looking at
`t["object"]`, which is wrong whenever the seed was matched as a triple's
*object* — routinely true with `reverse_edges: true`, since seed entities
are often claims/tags ("post-workout", "high protein snack") rather than
product names. Every hop-2 attempt was discarding exactly the new,
useful nodes (the actual products) and keeping only the seed itself
(already visited), so `12 PK` per query understated what the local index
could actually find by roughly 16x once fixed.

Re-measured with `scripts/sweep_max_hops.py` (3 questions x 3 runs):

| max_hops | PK triples (avg) | graph_traversal |
|---|---|---|
| 1 | 11.3 | 13.3ms |
| **2 (new default)** | **183.7** | 13.7ms |
| 3-5 | 183.7 (plateaus on `max_triples: 40`) | ~13.6ms |

`polysorbate 20`'s 0 -> 192 in the table above was already correct (it was
discovered as a *subject*, forward edges alone reach it); what this fix
adds is everything discovered only through claim/tag-style, object-matched
seeds, which forward-only and the original reverse-edges fix both missed.
Full detail in `PROGRESS.md`'s "2026-07-29: second GB10 pass".
