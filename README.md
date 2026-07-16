# Agentic Retrieval + Knowledge Graph + DFlash

![Agentic Retrieval overview](AgenticRetrievalOverview.png)

This repository extends the [AgenticRetrieval](https://github.com/bvonodiripsa/AgenticRetrieval) project with two major additions: a **Knowledge Graph (KG)** retrieval layer built on Azure Cosmos DB, and **DFlash speculative decoding** for GPU-accelerated LLM inference. Together they deliver faster, higher-quality answers over the same food product dataset (58K documents, 892K KG triples).

A unified FastAPI web application (`api.py`) exposes three backend pipelines side by side:

| Backend | LLM | Retrieval Strategy | Decoding |
|---------|-----|--------------------|----------|
| **Original** | GPT-4.1 (Azure OpenAI) | Multi-round decomposed RAG (vector + full-text) | Standard cloud |
| **KG-RAG** | Qwen3.5-27B (local vLLM, FP8) | KG graph traversal + vector + source fetch | Standard local |
| **KG-RAG + DFlash** | Qwen3.5-27B (local vLLM, FP8) | KG traversal + vector + keyword + semantic rerank | DFlash speculative |

## What Changed from the Original

### Architecture changes

| Area | Original ([AgenticRetrieval](https://github.com/bvonodiripsa/AgenticRetrieval)) | This repo (AgenticRetrieval-DFlash) |
|------|--------------------------|------|
| **LLM** | Azure OpenAI GPT-4.1 (cloud API) | Qwen3.5-27B served locally via vLLM (FP8 quantized) |
| **Hardware** | No GPU required (cloud LLM) | 2x NVIDIA H100 80GB (Azure ND96isr_H100_v5) |
| **Retrieval** | Multi-round decomposed RAG: sub-question decomposition, gap-filling re-retrieval over 2+ rounds | Single-pass KG traversal: entity search → graph hop → source fetch |
| **Knowledge Graph** | None — retrieves directly from document embeddings | 892K triples extracted from 58K documents; stored in Cosmos DB (`kg_triples_food`, `kg_entities_food`) |
| **Embedding** | Azure OpenAI embedding endpoint | In-process `Qwen3-Embedding-0.6B` (1024 dims, no network call) |
| **Decoding** | Standard autoregressive | DFlash speculative: `z-lab/Qwen3.5-27B-DFlash` draft model proposes 5 tokens per step; main model verifies in a single forward pass |
| **Semantic Reranker** | Optional (disabled by default) | Integrated via Cosmos DB Semantic Reranker SDK (DFlash path) |
| **Keyword Search** | Built-in full-text search per source | LLM-expanded keyword generation + parallel `FullTextContains` queries |
| **Web UI** | CLI only (`dynamic_retriever.py`) | FastAPI + SSE streaming web app with real-time progress and timing |

### New files (not in the original)

| File | Purpose |
|------|---------|
| `kg_builder.py` | Offline KG construction: triple extraction, dedup, predicate normalization, entity resolution |
| `kg_query.py` | Online KG-RAG query engine: graph traversal + vector augment + LLM answer |
| `api.py` | FastAPI web app serving the KG-RAG + LLM backend with SSE streaming |
| `prompts_kg_food.py` | KG-specific prompts for triple extraction and answer generation |
| `static/index.html` | Web UI with progress log and timing display |
| `config.yaml.example` | Consolidated config template (copy to `my.yaml`, then fill in secrets) |
| `upstream.py` + `scripts/sync_upstream.*` | Vendor the upstream AgenticRetrieval repo into `external/agenticretrieval` (git-ignored, re-syncable) |

## How the Pipelines Work

### Pipeline 1: Original AgenticRetrieval (decomposed multi-round RAG)

This is the upstream [AgenticRetrieval](https://github.com/bvonodiripsa/AgenticRetrieval) pipeline running unmodified. It is the quality baseline.

```
Question
  │
  ├─► Round 1: Vector + full-text search → initial answer
  │
  ├─► Gap analysis: identify missing knowledge
  │
  ├─► Round 2: Decompose gaps into sub-questions
  │      ├─► Sub-question 1 → targeted retrieval
  │      ├─► Sub-question 2 → targeted retrieval
  │      └─► ...
  │
  └─► Final synthesis: combine all evidence → answer
```

- **LLM**: GPT-4.1 via Azure OpenAI (cloud, ~30-40 tok/s)
- **Retrieval**: `CombinedRetriever` from `dynamic_retriever.py` — vector search (k=35) + full-text search (k=15) per source container, with diversity selection
- **Rounds**: 2 decompose/retrieve/synthesize rounds by default
- **Strengths**: Highest answer completeness (10+ products, detailed reasoning); gap-aware re-retrieval catches information missed in the first pass
- **Weakness**: Slowest — multiple LLM calls + multiple retrieval rounds (70-94s total)

### Pipeline 2: KG-RAG (knowledge graph retrieval + single LLM call)

Single-pass retrieval through a pre-built knowledge graph, followed by one LLM call.

```
Question
  │
  ├─► Embed question (Qwen3-Embedding-0.6B, in-process)
  │
  ├─► Entity search (vector search on kg_entities_food, top 20)
  │
  ├─► Graph traversal (PK-based triple fetch per entity, 2 hops)
  │      └─► Vector triple search (top 30 by similarity)
  │
  ├─► Source fetch (document IDs from triples/entities → batch read)
  │      └─► Vector augment (additional food container search, top 15)
  │
  └─► Single LLM call (Qwen3.5-27B, standard decoding) → answer
```

- **LLM**: Qwen3.5-27B via vLLM (local, FP8, ~55 tok/s standard)
- **KG**: 892K triples linking products, ingredients, allergens, dietary properties, occasions, cooking methods
- **Context**: Large window — up to 150 triples + 40 source chunks + 15 vector augment docs
- **Strengths**: Rich structured context from KG; single LLM call
- **Weakness**: Still relatively slow (~35-43s) because the LLM generates with standard autoregressive decoding over a large context

### Pipeline 3: KG-RAG + DFlash (parallel retrieval + speculative decoding)

Same KG retrieval as Pipeline 2 but with two key optimizations: **parallel retrieval** and **DFlash speculative decoding**.

```
Question
  │
  ├─► Embed question (Qwen3-Embedding-0.6B, in-process)
  │
  ├─► PARALLEL:
  │      ├─► Entity search (vector, top 10)
  │      └─► LLM keyword expansion (5-8 food-related search terms)
  │
  ├─► PARALLEL:
  │      ├─► Graph traversal (PK fetch + vector triples)
  │      ├─► Food vector search
  │      └─► Full-text keyword search (per expanded keyword)
  │
  ├─► Merge + deduplicate all results
  │
  ├─► Semantic reranker (Cosmos DB AI reranker, top 25)
  │
  └─► Single LLM call (Qwen3.5-27B + DFlash draft model) → answer
```

- **LLM**: Qwen3.5-27B via vLLM with DFlash speculative decoding (~110-140 tok/s, 2-2.5x speedup)
- **Retrieval**: All search paths run concurrently via `asyncio.gather`; reduced context limits (40 triples, 15 source chunks) to minimize prompt tokens
- **Keyword expansion**: LLM generates additional food-related search terms (e.g., "protein bar", "energy", "peanut") run as parallel `FullTextContains` queries
- **Semantic reranker**: Cosmos DB Semantic Reranker re-orders retrieved documents by relevance before prompting the LLM
- **Strengths**: Fastest pipeline (17-23s); lossless quality (DFlash output is mathematically identical to standard decoding)
- **Context limits vs KG**: Smaller context (1200 vs 2048 answer tokens, 40 vs 150 triples) trades some breadth for speed

## How DFlash Speculative Decoding Works

DFlash is a speculative decoding technique that accelerates LLM inference without changing the output distribution.

```
Standard decoding (1 token per forward pass):
  Step 1: [prompt] → token_1
  Step 2: [prompt, token_1] → token_2
  Step 3: [prompt, token_1, token_2] → token_3
  ... (N forward passes for N tokens)

DFlash speculative decoding (up to 6 tokens per forward pass):
  Step 1: Draft model proposes [d1, d2, d3, d4, d5]  (cheap, ~1B params)
  Step 2: Main model verifies all 5 in ONE forward pass
  Step 3: Accept first K correct tokens, reject the rest
  Step 4: Repeat from the first rejected position
```

- **Draft model**: `z-lab/Qwen3.5-27B-DFlash` — a small model (~1B params) trained to mimic Qwen3.5-27B's token distribution
- **Verification**: The main Qwen3.5-27B model checks all draft tokens in a single batched forward pass
- **Acceptance rate**: Typically 60-80% of draft tokens are accepted, yielding 2-2.5x effective throughput
- **Lossless**: The rejection-sampling scheme guarantees the output distribution is identical to standard autoregressive generation

### vLLM configuration

```bash
vllm serve Qwen/Qwen3.5-27B \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92 \
  --dtype float16 \
  --quantization fp8 \
  --spec-model z-lab/Qwen3.5-27B-DFlash \
  --spec-tokens 5 \
  --enable-prefix-caching \
  --port 8000
```

## How to Build the Knowledge Graph

The KG is built offline using `kg_builder.py`. It reads food product documents from Cosmos DB, extracts structured triples via LLM, post-processes them, and stores the KG back to Cosmos DB.

### KG build pipeline

1. **Read documents** from the `food` container in Cosmos DB (all 58K or a question-driven subset via vector search)
2. **Extract triples** using Qwen3.5-27B with decomposed extraction:
   - Round 1: Initial extraction from product fields (title, ingredients, claims, nutrition)
   - Round 2+: Gap analysis identifies missing knowledge, targeted extraction fills gaps
3. **Dedup + confidence boost**: Merge duplicate triples; boost confidence when triples are re-confirmed across documents
4. **Normalize predicates**: LLM batches standardize free-form predicates into a controlled vocabulary (`has_ingredient`, `contains_allergen`, `suitable_for_occasion`, `has_cooking_method`, etc.)
5. **Entity resolution**: Embedding-based clustering (cosine similarity > 0.85) + LLM merge verification to unify variant entity names
6. **Store to Cosmos DB**: Upsert triples and entities with embeddings to `kg_triples_food` and `kg_entities_food` containers

### CLI usage

```bash
# Full KG build
python kg_builder.py --config my.yaml

# Question-driven subset (faster for testing)
python kg_builder.py --config my.yaml --question-driven --question-k 30

# Resume from checkpoint
python kg_builder.py --config my.yaml --time-limit 3600

# Skip extraction, only run post-processing
python kg_builder.py --config my.yaml --skip-extraction --reprocess
```

### Triple schema in Cosmos DB

```json
{
  "id": "triple-hash",
  "pk": "reeses sticks",
  "subject": "Reese's Sticks",
  "predicate": "has_ingredient",
  "object": "peanut butter",
  "confidence": 0.95,
  "confirmations": 3,
  "source_chunks": ["doc-abc123"],
  "embedding": [0.012, -0.034, ...]
}
```

### Inferred semantic triples

Beyond extracting facts directly from product data, the builder infers higher-level semantic triples:

- **Occasions**: breakfast, snack, dessert, BBQ, cinema, picnic
- **Cooking methods**: ready to eat, microwave, air fryer, grill, oven
- **Convenience**: instant, under 5 min, under 15 min, requires cooking
- **Nutrition**: high protein, high calorie, low sugar, low fat, keto-friendly
- **Portability**: pocket sized, single serving, family size
- **Audience**: athletes, health conscious, families, children

## Configuration Reference

The app is driven by a single YAML config: copy `config.yaml.example` to `my.yaml`
(git-ignored) and fill in your Cosmos DB, embedding, and LLM settings + secrets.
Override the path with `--config <file>`.

### Key query settings (`query:` block)

| Setting | Value | Purpose |
|---------|-------|---------|
| `seed_entities_k` | 10 | Seed entities from vector search |
| `max_hops` | 1 | Graph traversal depth |
| `max_triples` | 40 | Triples passed to the LLM |
| `max_source_chunks` | 15 | Source documents fetched |
| `vector_augment_k` | 12 | Extra vector-search products |
| `max_answer_tokens` | 4096 | Answer budget (covers reasoning tokens) |

Embeddings are computed in-process (Qwen3-Embedding-0.6B, mean-pool + L2). The
Cosmos DB semantic reranker reorders sources before the LLM call; if it is
unavailable the pipeline falls back to vector-search ordering. Keyword search is
LLM-expanded via `FullTextContains`.

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| **GPU** | 2x NVIDIA H100 80GB (for vLLM with tensor parallelism) |
| **VM** | Azure ND96isr_H100_v5 (96 vCPU, 1.9TB RAM) |
| **Disk** | 256GB+ for model weights and checkpoints |
| **Network** | Azure VNet with NSG rules for port 8080 (web UI) |

The Original backend (GPT-4.1) requires no local GPU — it calls Azure OpenAI APIs. However, all three backends share the same Azure Cosmos DB account and in-process embedding model.

### vLLM memory layout (2x H100)

| Component | Memory |
|-----------|--------|
| Qwen3.5-27B weights (FP8) | ~27 GB across 2 GPUs |
| DFlash draft model | ~1 GB |
| KV cache | ~50 GB (FP8, 16K context) |
| CUDA overhead | ~10 GB |
| **Total** | ~88 GB / 160 GB available |

## Benchmark Results

**Hardware**: 2x NVIDIA H100 NVL (Sweden Central) | **Database**: Cosmos DB (Sweden Central, co-located) — 58K food products, 1.6M triples, 180K entities

### Per-question timing (Q1: "high-calorie protein snack for running belt")

| Stage | Time |
|-------|------|
| Embed | 0.27s |
| Entity Search | 0.33s |
| Graph Traversal | 0.83s |
| Source Fetch | 0.26s |
| LLM (DFlash) | 7.71s |
| **Total** | **9.40s** |

### Full pipeline (web app, warm)

| Stage | Time |
|-------|------|
| Embed | 0.30s |
| Entity Search | 0.84s |
| Graph Traversal (parallel) | 1.40s |
| Source Fetch | 0.40s |
| LLM (DFlash) | 3.46s |
| **Total** | **6.40s** |

### Region co-location impact

Co-locating Cosmos DB and the VM in the same Azure region reduced retrieval latency by 5-10x:

| Query | Cross-region | **Co-located** | Speedup |
|-------|-------------|---------------|---------|
| Baseline (no vector) | 0.50s | **0.10s** | 5x |
| Entity vector (180K docs) | 0.51s | **0.30s** | 1.7x |
| Triple vector (1.6M docs) | 0.52s | **0.35s** | 1.5x |

### Quality

- DFlash speculative decoding is **mathematically lossless** — output quality is identical to standard Qwen3.5-27B generation
- Keyword expansion and semantic reranking improve product coverage
- Answers include 5-10 product recommendations with reasoning

## Running the Web Application

### Prerequisites

1. Azure Cosmos DB account with `food`, `kg_entities_food`, `kg_triples_food` containers populated
2. vLLM server running on port 8000 (see vLLM configuration above)
3. Azure CLI logged in (`az login`) for Cosmos DB RBAC
4. Semantic reranker endpoint set in the config (`cosmos.semantic_reranker_endpoint`); an `AZURE_COSMOS_SEMANTIC_RERANKER_INFERENCE_ENDPOINT` env var overrides it

### Start the web app

```bash
pip install -r requirements-web.txt

# The app reads a single config (default my.yaml; override with --config).
# The Cosmos reranker endpoint comes from cosmos.semantic_reranker_endpoint.
python api.py --config my.yaml --host localhost --port 8080
```

To launch with uvicorn directly (e.g. to pass extra uvicorn flags), set the
config via the `KG_CONFIG` environment variable instead:

```bash
KG_CONFIG=my.yaml \
  python -m uvicorn api:app --host localhost --port 8080 --timeout-keep-alive 120
```

### API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/v1/backends` | GET | Available backends with descriptions |
| `/v1/questions` | GET | Benchmark questions |
| `/v1/ask/stream` | POST | SSE streaming answer (`{"question": "..."}`) |
| `/v1/ask` | POST | JSON response (non-streaming) |

### SSE event format

```
data: {"stage": "progress", "message": "Embedding question...", "_ts": 0.0}
data: {"stage": "progress", "message": "Found 10 entities in 1.3s", "_ts": 1.3}
data: {"stage": "stats", "entities": 10, "triples": 40, "sources": 65}
data: {"stage": "answer_chunk", "text": "Based on the provided data..."}
data: {"stage": "done", "timings": {"embed": 0.3, "entity_search": 1.3, ...}}
```

## Azure Cosmos DB Semantic Reranker

The DFlash pipeline integrates the [Cosmos DB Semantic Reranker](https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/semantic-reranker) to re-order retrieved documents by semantic relevance before passing them to the LLM.

### Setup

1. Enable the Semantic Reranker on your Cosmos DB account via the Azure portal
2. Register the provider: `az provider register -n Microsoft.InferenceService`
3. Assign the "Semantic Reranker User" role on the **InferenceService** resource:
   ```bash
   az role assignment create \
     --role "Semantic Reranker User" \
     --assignee-object-id "<your-user-object-id>" \
     --assignee-principal-type "User" \
     --scope "/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.InferenceService/inferenceAccounts/<account>"
   ```
4. Set the endpoint in your config (recommended):
   ```yaml
   cosmos:
     semantic_reranker_endpoint: "https://<account>.<region>.dbinference.azure.com"
   ```
   Or export an env var to override the config:
   ```bash
   export AZURE_COSMOS_SEMANTIC_RERANKER_INFERENCE_ENDPOINT="https://<account>.<region>.dbinference.azure.com"
   ```

The reranker is called after retrieval and before the LLM, reordering source documents by relevance to the question. If the reranker call fails (e.g., RBAC not configured), the pipeline falls back to vector-search ordering.

## Repository Layout

```
AgenticRetrieval-DFlash/
├── api.py                      # FastAPI web app (single KG + LLM backend)
├── kg_builder.py               # Offline KG construction
├── kg_query.py                 # Online KG-RAG query engine
├── prompts_kg_food.py          # KG-specific prompts
├── upstream.py                 # Bootstrap for the vendored upstream clone
├── static/index.html           # Web UI
├── config.yaml.example         # Consolidated config template (copy to my.yaml)
├── external/agenticretrieval/  # Vendored upstream (git-ignored; sync_upstream.*)
├── data/food.json              # 10 benchmark questions
├── ARCHITECTURE.md             # Detailed code-level architecture
├── BENCHMARKS.md               # Timing benchmark tables
├── dynamic_retriever.py        # Original decomposed RAG (shared with upstream)
├── cosmos_db_upload.py         # Document ingestion
├── requirements-web.txt        # Web app dependencies
├── requirements.txt            # Full dependencies
└── out_kg_dflash/              # DFlash benchmark outputs
```

## License

MIT — see [LICENSE](LICENSE).
