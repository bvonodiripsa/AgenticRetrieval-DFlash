# Architecture — Multi-Backend Food KG-RAG

This document describes the code paths for each of the 3 backends in the Food KG-RAG application.

## Overview

| Backend | LLM | Retrieval | Decoding | Avg Time |
|---------|-----|-----------|----------|----------|
| **Original** | GPT-4.1 (Azure OpenAI) | Multi-round decomposed RAG | Standard | ~70-94s |
| **KG-RAG** | Qwen3.5-27B (local, FP8) | Knowledge Graph traversal | Standard | ~35-43s |
| **KG-RAG + DFlash** | Qwen3.5-27B (local, FP8) | KG traversal + keyword expansion | DFlash speculative | ~17-27s |

---

## 1. Original AgenticRetrieval (GPT-4.1)

**Routing**: `api.py` → `_run_original_stream()`

**Code flow**:
- `_init_original()` in `api.py` dynamically imports `DecomposedRAGPipeline` from `/home/azureuser/AgenticRetrieval/dynamic_retriever.py`
- Loads config from `config_original_local.yaml` — points to Azure OpenAI GPT-4.1
- `_original_pipeline.run(question)` runs the full multi-round decomposed RAG:
  - **Round 1**: Decomposes question into sub-queries
  - **Rounds 2-5**: Each round does vector + full-text search via `CombinedRetriever`, generates intermediate answers, refines
  - Uses `LLMClient` — calls Azure OpenAI GPT-4.1 with RBAC auth
- `_run_original_stream()` replays the final answer via SSE in 20-char chunks

**Key files** (external repo `/home/azureuser/AgenticRetrieval/`):
- `dynamic_retriever.py` — `DecomposedRAGPipeline`
- `utils/cosmos_retriever.py` — `CombinedRetriever`
- `utils/llm_client.py` — `LLMClient` (Azure OpenAI)

**Config**: `config_original_local.yaml`

---

## 2. KG-RAG (Qwen3.5-27B, standard decoding)

**Routing**: `api.py` → `_stream_kg_sse(question, engine, backend_id="kg")`

**Code flow** (`api.py`):
1. `engine._embedder.embed()` — Qwen3-Embedding-0.6B (in-process)
2. Entity vector search on `kg_entities_food` container
3. **PK-based graph traversal** — for each seed entity, fetch triples by partition key, 2 hops
4. **Vector triple search** — top-30 triples by embedding similarity
5. Deduplicate triples
6. **Source chunk fetch** — collect document IDs from triples/entities, fetch from `food` container by ID
7. **Vector augment** — additional vector search on `food` container, merge
8. Build prompt using `GRAPHRAG_ANSWER_PROMPT` from `prompts_kg_food.py`
9. **Non-streaming LLM call** to Qwen3.5-27B via vLLM (localhost:8000), standard autoregressive decoding
10. Replay answer in 80-char chunks via SSE

**Key files**:
- `api.py` — `_stream_kg_sse()`
- `kg_query.py` — `KGQueryEngine` (Cosmos client, embedder, `_build_graph_context()`, `_build_source_text()`)
- `prompts_kg_food.py` — `GRAPHRAG_ANSWER_PROMPT`

**Config**: `config_kg_oldqwen.yaml`

---

## 3. KG-RAG + DFlash (Qwen3.5-27B, speculative decoding)

**Routing**: `api.py` → `_stream_dflash_sse(question, engine)`

**Code flow** (`api.py`):
1. `engine._embedder.embed()` — Qwen3-Embedding-0.6B (in-process)
2. **Entity search + LLM keyword expansion** — parallel via `asyncio.gather`
   - Entity vector search on `kg_entities_food`
   - `_llm_expand_keywords()` — lightweight LLM call to extract food search terms (e.g., "peanut", "energy bar")
3. **Parallel retrieval** via `asyncio.gather`:
   - `_graph_traversal()` — PK-based hop traversal (same as KG)
   - `_triple_vec()` — vector triple search
   - `_food_vec()` — vector food search
   - `_food_fulltext()` × N — per-keyword `FullTextContains` queries
4. Deduplicate triples (PK + vector)
5. **Source chunk fetch** — collect IDs from triples/entities, fetch by ID
6. **Merge** — KG sources + vector + keyword results, deduplicate
7. `_semantic_rerank()` — Cosmos DB semantic reranker (pending RBAC, falls back to original order)
8. Build prompt using `DFLASH_ANSWER_PROMPT` (defined in `api.py`)
9. **Non-streaming LLM call** to Qwen3.5-27B via vLLM — **DFlash speculative decoding** (draft model `z-lab/Qwen3.5-27B-DFlash` proposes 15 tokens, target model verifies in batch)
10. Replay answer in 80-char chunks via SSE

**Key files**:
- `api.py` — `_stream_dflash_sse()`, `_dflash_answer()`, `_llm_expand_keywords()`, `_extract_keywords()`, `_semantic_rerank()`
- `kg_query.py` — `KGQueryEngine`
- `prompts_kg_food.py` — prompt templates

**Config**: supplied via the required `--config` argument — e.g. `config_kg_dflash.yaml` (local Qwen3.5-27B + DFlash) or `config_kg_glm.yaml` (hosted GLM-5.2)

---

## What's Different Between KG and DFlash?

| Aspect | KG | DFlash |
|--------|-----|--------|
| Retrieval | Sequential steps | Parallel `asyncio.gather` |
| Keyword search | None | LLM-expanded + `FullTextContains` |
| Semantic reranker | No | Yes (pending RBAC) |
| LLM decoding | Standard autoregressive | DFlash speculative (2-2.5x faster) |
| Prompt | `GRAPHRAG_ANSWER_PROMPT` | `DFLASH_ANSWER_PROMPT` (asks for 8-10 products) |

## How DFlash Speculative Decoding Works

The application code makes a standard OpenAI-compatible API call — the speculative decoding is handled entirely by vLLM:

1. **Draft model** (`z-lab/Qwen3.5-27B-DFlash`, ~1-2B params) generates 15 candidate tokens cheaply
2. **Target model** (Qwen3.5-27B, 27B params) verifies all 15 in a single forward pass
3. Accepted tokens are kept; rejected tokens are replaced
4. Repeat until generation is complete

This produces **identical output** to standard generation (mathematically lossless) while requiring ~2-3x fewer expensive forward passes through the full model.

## Shared Infrastructure

- **Cosmos DB**: `divdet` account, `food` database — containers: `food` (58K products), `kg_entities_food`, `kg_triples_food`
- **vLLM server**: `localhost:8000`, Qwen3.5-27B (FP8), tensor-parallel 2x H100
- **Embedding**: Qwen3-Embedding-0.6B, loaded in-process via `sentence-transformers`
- **Web UI**: `static/index.html`, FastAPI on port 8080
