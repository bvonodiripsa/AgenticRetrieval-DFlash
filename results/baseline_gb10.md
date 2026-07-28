# GB10 baseline — existing code, unmodified

Recorded 2026-07-27. Reference point for the local-index optimization.

## Setup

| | |
|---|---|
| Machine | NVIDIA GB10, aarch64, driver 580.173.02, CUDA 13.0 |
| Client location | US west coast |
| Cosmos DB | `divdet-sweden` (Sweden Central) — cross-continental |
| Embedder | `Qwen/Qwen3-Embedding-0.6B` in-process, **CPU** (no `.to("cuda")`) |
| LLM | none running; stage not measured |
| Harness | `benchmark_pipeline.py`, 3 consecutive runs |
| Config | `my.yaml` — `seed_entities_k: 10`, `max_hops: 1`, `max_triples: 40`, `max_source_chunks: 15`, `vector_augment_k: 12` |

Container sizes verified against the live account:

| Container | Documents | Vector dim |
|---|---|---|
| entities | 179,560 | 1024 |
| triples | 1,593,678 | 1024 |
| food | 58,233 | 1024 |

## Results

| Stage | Run 1 | Run 2 | Run 3 | Mean | H100 co-located (README) |
|---|---|---|---|---|---|
| Embed | 1.27s | 1.30s | 1.29s | **1.29s** | 0.30s |
| Entity Search | 8.20s | 8.13s | 8.17s | **8.17s** | 0.84s |
| Graph Traversal | 12.44s | 11.72s | 11.64s | **11.93s** | 1.40s |
| Source Fetch | 4.30s | 4.24s | 4.11s | **4.22s** | 0.40s |
| **Retrieval total** | 26.21s | 25.39s | 25.21s | **25.60s** | **2.94s** |

Retrieval from GB10 is 8.7x slower than the co-located H100 figure. The gap is
network distance to Sweden Central, not GB10 compute — the same 14 Cosmos
queries are issued in both cases.

## Graph traversal returns nothing

Every run reports `0 PK + 30 vec`. No triples come from graph traversal; all 30
come from the triple vector search.

Cause: `_triples_pk_field` is `s`, and triple subjects are product names.
Entity vector search returns ingredient-level entities, which appear in the
graph only as objects.

```
c.s = 'royal dansk oatmeal cookies with cranberries (product_id: 22080918)' -> 30 triples
c.s = 'polysorbate 20'                                                      ->  0 triples
c.s = 'sweetened dried cranberry (cranberries, sugar)'                      ->  0 triples
```

Traversal follows subject to object only, so ingredient seeds match nothing.
The Graph Index is currently behaving as a pure vector index.

Fixing this in Cosmos needs `WHERE c.s = @pk OR c.o = @pk`. Since `c.o` is not
the partition key that is a cross-partition scan on 1.59M documents. An
in-memory CSR index stores forward and reverse adjacency at no extra cost.

## Note for the comparison

The optimized run should be measured twice: once with traversal still
forward-only to keep the comparison honest, and once with reverse edges enabled
to show what the fix is worth.
