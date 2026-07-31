# Architecture — Food GI-RAG (single Graph Index + LLM backend)

This document describes the code path for the Food GI-RAG application: a single
graph-index retrieval + LLM answer pipeline. The upstream AgenticRetrieval
decomposed-RAG code is vendored (git-ignored) under `external/agenticretrieval`
(see `scripts/sync_upstream.*`) and used only by the `samples/QA_CLI` demo and
the tests — not by this app.

## Overview

| LLM | Retrieval | Decoding |
|-----|-----------|----------|
| Configurable (local vLLM, or a hosted OpenAI-compatible endpoint such as GLM-5.2) | Graph Index traversal + vector + LLM keyword expansion + semantic rerank | Speculative decoding when the model/endpoint supports it |

---

## GI-RAG + LLM pipeline

**Routing**: `api.py` → `_stream_dflash_sse()` (streaming, `/v1/ask/stream`) /
`_dflash_answer()` (non-streaming, `/v1/ask`)

**Code flow** (`api.py`):
1. `engine._embedder.embed(question, is_query=True)` — Qwen3-Embedding-0.6B (in-process, last-token pool + L2, query instruction prefix)
2. **Retrieval + LLM keyword expansion** — parallel via `asyncio.gather`:
   - `retrieval.retrieve(backend, q_emb, cfg)` — entity search, graph
     traversal, triple/food vector search and source fetch, all against
     whichever backend `index.mode` selected (see below)
   - `_llm_expand_keywords()` — lightweight LLM call to extract food search terms
3. **Keyword full-text merge** — `backend.fulltext_food(keywords)`, deduped
   into `retrieve()`'s source chunks
4. `_semantic_rerank()` — Cosmos DB semantic reranker (falls back to vector order)
5. Build prompt using `DFLASH_ANSWER_PROMPT` (defined in `api.py`)
6. **Single LLM call** via the configured OpenAI-compatible endpoint, with
   `stream=True` on the streaming path (real per-token SSE, not
   buffer-then-chunk); reasoning/thinking suppression is chosen per model
   family by `gi_query.build_llm_call_kwargs()` (Qwen `enable_thinking=false`;
   reasoning models get `reasoning_effort` when set)

**Key files**:
- `api.py` — `_stream_dflash_sse()`, `_dflash_answer()`, `_llm_expand_keywords()`, `_extract_keywords()`, `_semantic_rerank()`
- `retrieval.py` — `retrieve()`, the single retrieval implementation shared by both backends and both endpoints (streaming and non-streaming; also reused by `gi_query.py`'s CLI/benchmark path)
- `gi_query.py` — `GIQueryEngine`, `_get_backend()` (backend selection/hot-swap), `build_llm_call_kwargs()`, `_build_graph_context()`, `_build_source_text()`
- `gi_index.py` — `LocalGraphIndex`, the GPU-resident backend (see below)
- `prompts_gi_food.py` — prompt templates

**Config**: a single YAML (default `my.yaml`, from `config.yaml.example`; override with `--config`).

---

## Retrieval backends: Cosmos vs. local GPU index

`retrieval.py::retrieve()` is written once against a small backend
interface (`entity_search`, `triples_for`, `triples_vec`, `food_vec`,
`fetch_by_ids`, `fulltext_food`) and works unchanged against either
implementation. `index.mode` in the config picks which one:

| | `index.mode: cosmos` (`CosmosBackend`, `gi_query.py`) | `index.mode: local` (`LocalBackend` / `LocalGraphIndex`, `gi_index.py`) |
|---|---|---|
| Where data lives | Azure Cosmos DB (network) | GPU memory (vectors) + host RAM (strings, CSR graph) |
| Vector search | Cosmos DiskANN (quantized, approximate) | Exact brute-force `torch.topk` (unquantized, more accurate at this corpus size) |
| Graph traversal | Per-hop Cosmos query, partition-key (`subject`) lookup only | CSR adjacency array; `reverse_edges: true` also traverses `object -> subject` for free, which Cosmos cannot do cheaply since `object` isn't the partition key |
| Full-text search | `FullTextContains` (Cosmos) | Local BM25 (`rank_bm25`) over the food container |
| Measured cost (co-located H100, warm) | ~2.06-2.26s | ~0.003-0.03s |

Cosmos remains the **system of record**; the local index is a read-only
snapshot exported by `scripts/build_local_index.py` and must be rebuilt
whenever the Graph Index changes upstream. It needs ~3.8 GB of GPU memory
for the full corpus (fits alongside vLLM's KV cache).

**Backend selection and hot-swap** (`gi_query.py::GIQueryEngine._get_backend()`):
- `mode: "local"` + snapshot exists at `index.snapshot_path` → loads
  `LocalGraphIndex` (numpy → GPU, BM25 build) once, eagerly, at API startup
  (`api.py`'s `lifespan()`), not on the first request.
- `mode: "local"` + no snapshot yet + `auto_build: true` → serves
  `CosmosBackend` immediately (cheap, no blocking network call) and kicks
  off `scripts/build_local_index.py`'s exporter as a background
  `asyncio.Task`; once it finishes, swaps `self._backend` to
  `LocalGraphIndex` without downtime or a restart.
- `check_freshness: true` fires a non-blocking background comparison of the
  snapshot's manifest against live Cosmos counts/timestamps
  (`snapshot_freshness.py`) on every backend load, logging a warning if
  stale. This is a signal for an operator, not an automatic rebuild trigger.

---

## How speculative decoding works

The application code makes a standard OpenAI-compatible API call — the speculative decoding is handled entirely by vLLM:

1. **Draft model** (`z-lab/Qwen3.5-27B-DFlash`, ~1-2B params) generates 15 candidate tokens cheaply
2. **Target model** (Qwen3.5-27B, 27B params) verifies all 15 in a single forward pass
3. Accepted tokens are kept; rejected tokens are replaced
4. Repeat until generation is complete

This produces **identical output** to standard generation (mathematically lossless) while requiring ~2-3x fewer expensive forward passes through the full model.

## Shared Infrastructure

- **Cosmos DB**: account + `food` database from config — containers: `food` (products), `entities`, `triples`; system of record regardless of `index.mode`
- **Local Graph Index** (optional, `index.mode: local`): GPU-resident snapshot of the same data — see "Retrieval backends" above and `PROGRESS.md`
- **LLM endpoint**: configurable OpenAI-compatible (local vLLM with DFlash, or a hosted gateway such as GLM-5.2)
- **Embedding**: Qwen3-Embedding-0.6B, loaded in-process (last-token pool + L2, query instruction prefix, GPU if available; see `EMBEDDING_FIX.md`)
- **Web UI**: `static/index.html`, FastAPI on port 8080
- **Vendored upstream**: `external/agenticretrieval` (git-ignored) via `scripts/sync_upstream.*`
