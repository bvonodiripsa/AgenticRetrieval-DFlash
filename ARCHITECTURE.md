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

**Routing**: `api.py` → `_stream_dflash_sse()` (streaming) / `_dflash_answer()` (non-streaming)

**Code flow** (`api.py`):
1. `engine._embedder.embed()` — Qwen3-Embedding-0.6B (in-process, mean-pool + L2)
2. **Entity search + LLM keyword expansion** — parallel via `asyncio.gather`
   - Entity vector search on `entities`
   - `_llm_expand_keywords()` — lightweight LLM call to extract food search terms
3. **Parallel retrieval** via `asyncio.gather`:
   - `_graph_traversal()` — PK-based hop traversal
   - `_triple_vec()` — vector triple search
   - `_food_vec()` — vector food search
   - `_food_fulltext()` × N — per-keyword `FullTextContains` queries
4. Deduplicate triples (PK + vector)
5. **Source chunk fetch** — collect IDs from triples/entities, fetch by ID
6. **Merge** — GI sources + vector + keyword results, deduplicate
7. `_semantic_rerank()` — Cosmos DB semantic reranker (falls back to vector order)
8. Build prompt using `DFLASH_ANSWER_PROMPT` (defined in `api.py`)
9. **Single LLM call** via the configured OpenAI-compatible endpoint; reasoning /
   thinking suppression is chosen per model family by `gi_query.build_llm_call_kwargs()`
   (Qwen `enable_thinking=false`; reasoning models get `reasoning_effort` when set)
10. Stream/replay the answer in 80-char chunks via SSE

**Key files**:
- `api.py` — `_stream_dflash_sse()`, `_dflash_answer()`, `_llm_expand_keywords()`, `_extract_keywords()`, `_semantic_rerank()`
- `gi_query.py` — `GIQueryEngine`, `build_llm_call_kwargs()`, `_build_graph_context()`, `_build_source_text()`
- `prompts_gi_food.py` — prompt templates

**Config**: a single YAML (default `my.yaml`, from `config.yaml.example`; override with `--config`).

---

## How speculative decoding works

The application code makes a standard OpenAI-compatible API call — the speculative decoding is handled entirely by vLLM:

1. **Draft model** (`z-lab/Qwen3.5-27B-DFlash`, ~1-2B params) generates 15 candidate tokens cheaply
2. **Target model** (Qwen3.5-27B, 27B params) verifies all 15 in a single forward pass
3. Accepted tokens are kept; rejected tokens are replaced
4. Repeat until generation is complete

This produces **identical output** to standard generation (mathematically lossless) while requiring ~2-3x fewer expensive forward passes through the full model.

## Shared Infrastructure

- **Cosmos DB**: account + `food` database from config — containers: `food` (products), `entities`, `triples`
- **LLM endpoint**: configurable OpenAI-compatible (local vLLM with DFlash, or a hosted gateway such as GLM-5.2)
- **Embedding**: Qwen3-Embedding-0.6B, loaded in-process (mean-pool + L2)
- **Web UI**: `static/index.html`, FastAPI on port 8080
- **Vendored upstream**: `external/agenticretrieval` (git-ignored) via `scripts/sync_upstream.*`
