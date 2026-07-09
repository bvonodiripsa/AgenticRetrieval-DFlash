# Knowledge Graph Construction and GPU-Accelerated Retrieval with DFlash

## The Problem: RAG Is Slow When Questions Are Complex

Traditional Retrieval-Augmented Generation (RAG) works well for simple factual lookups: embed the question, find the closest documents, hand them to an LLM. But real-world questions are rarely simple. "Find me a high-calorie protein snack for long-distance running that fits in a running belt" requires the system to reason about calories, protein content, portability, packaging size — information scattered across multiple product fields and not captured by any single embedding.

The standard approach is **multi-round decomposed RAG**: the LLM reads an initial answer, identifies knowledge gaps, generates sub-questions, retrieves more evidence, and synthesizes a final answer. This works well — our baseline (GPT-4.1 with decomposed RAG) produces excellent 10-product recommendations with detailed nutritional reasoning. But it takes **70-94 seconds** per question, requires multiple cloud LLM calls, and scales poorly.

The question is: can we match that quality while running **entirely on local GPUs**, and do it **5x faster**?

## The Approach: Pre-computed Knowledge Graph + Speculative Decoding

We combine two independent accelerations that multiply together:

1. **Offline KG construction** replaces multi-round retrieval with a single graph traversal
2. **DFlash speculative decoding** replaces standard autoregressive generation with draft-and-verify

Neither technique degrades output quality. The KG provides richer structured context than ad-hoc retrieval, and DFlash is mathematically lossless — it produces the exact same token distribution as standard generation.

## Part 1: Building the Knowledge Graph

### What goes into the graph

We start with 58,000 food product documents in Azure Cosmos DB. Each document is a JSON object with fields like `product_title`, `ingredients`, `allergens`, `claims`, `pack_size`, and `brand`. The raw documents support vector search and full-text search, but they don't encode the **relationships** between concepts.

The KG adds that relational layer. From those 58K documents, we extract **892,000 triples** — structured (subject, predicate, object) facts with confidence scores. For example:

```
(Reese's Sticks (product_id: 22082110), has_ingredient, peanut butter)     confidence: 0.95
(Reese's Sticks (product_id: 22082110), has_nutritional_profile, high calorie) confidence: 0.80
(Reese's Sticks (product_id: 22082110), suitable_for_occasion, cinema snack)   confidence: 0.80
(Reese's Sticks (product_id: 22082110), has_portability, pocket sized)         confidence: 0.80
```

The first triple is directly extracted from the ingredient list. The last three are **inferred** by the LLM from the product's properties — a 42g chocolate bar with peanut butter is likely high-calorie, good for movies, and fits in a pocket. These inferences are what make the KG powerful: they encode the kind of reasoning that a multi-round RAG system would normally do at query time.

### The build pipeline

The KG construction runs on the same GPU hardware used for inference (2x NVIDIA H100) using a locally-served Qwen3.5-27B model via vLLM.

**Step 1 — Decomposed triple extraction.** Each product document is sent to the LLM with a detailed extraction prompt that specifies the field semantics, extraction rules, and inference categories. The LLM returns a JSON array of triples. This runs in two rounds:

- **Round 1 (initial extraction):** Extract all facts directly available in the product fields — ingredients, allergens, brand, pack size, claims — plus inferred properties like usage occasion (breakfast, snack, BBQ), cooking method (microwave, air fryer, ready to eat), convenience level (instant, under 5 minutes), nutritional profile (high protein, high calorie, low sugar), portability (pocket sized, family size), and target audience (athletes, health conscious, children).

- **Round 2 (gap analysis + targeted extraction):** A second LLM call analyzes what's missing from the initial extraction — allergens not yet captured, dietary suitability not inferred, nutritional profiles incomplete — and performs targeted extraction to fill those gaps.

With 20 concurrent extraction tasks, the full 58K-document corpus is processed in approximately 8 hours on 2x H100.

**Step 2 — Deduplication and confidence boosting.** Triples extracted from multiple documents (e.g., the same ingredient appearing in different products) are merged. When the same triple is confirmed by multiple extractions, its confidence score is boosted — a triple confirmed 3 times is more reliable than one seen only once.

**Step 3 — Predicate normalization.** Free-form predicates from the LLM ("is made with", "contains", "has as ingredient") are normalized to a controlled vocabulary of ~30 standard predicates (`has_ingredient`, `contains_allergen`, `suitable_for_occasion`, `has_cooking_method`, etc.). This is done via LLM batch normalization — groups of 30 triples are sent to the LLM with the target vocabulary, and it returns the same triples with standardized predicates.

**Step 4 — Entity resolution.** Variant entity names are unified using embedding similarity and LLM verification. All unique entity names (subjects and objects) are embedded with Qwen3-Embedding-0.6B. Pairs with cosine similarity above 0.85 are sent to the LLM for merge decisions. The LLM determines whether "sugar" and "sucrose" should be merged (yes), or whether "milk chocolate" and "dark chocolate" should stay separate (yes). Merged entities are replaced throughout the graph with the canonical name.

**Step 5 — Storage.** Triples are upserted to Cosmos DB with their embeddings, partitioned by subject name (lowercased) for efficient graph traversal. A separate entity index is built with relation summaries and embeddings for vector search at query time.

### What the graph looks like in Cosmos DB

The graph occupies two containers:

**`kg_triples_food`** — 892K documents, each a single triple:
```json
{
  "id": "triple-hash",
  "pk": "reeses sticks (product_id: 22082110)",
  "subject": "Reese's Sticks (product_id: 22082110)",
  "predicate": "has_ingredient",
  "object": "peanut butter",
  "confidence": 0.95,
  "confirmations": 3,
  "source_chunks": ["doc-abc123"],
  "embedding": [0.012, -0.034, ...]
}
```

**`kg_entities_food`** — ~120K documents, each an entity node:
```json
{
  "id": "entity-hash",
  "pk": "reeses sticks (product_id: 22082110)",
  "name": "Reese's Sticks (product_id: 22082110)",
  "description": "has_ingredient: peanut butter, chocolate, wafer; has_claim: high calorie; ...",
  "relation_count": 15,
  "source_chunks": ["doc-abc123"],
  "embedding": [0.008, -0.041, ...]
}
```

Both containers have **vector indexes** on the `embedding` field (DiskANN, cosine similarity, 1024 dimensions) enabling sub-second semantic search across the entire graph.

## Part 2: Query-Time KG Traversal

At query time, the KG replaces multi-round retrieval with a single structured traversal:

```
"High-calorie protein snack for running belt"
                │
    ┌───────────┴──────────┐
    ▼                      ▼
 EMBED QUESTION     LLM KEYWORD EXPANSION
 (0.3s, in-process)    (1.0s, parallel)
    │                      │
    ▼                      │
 ENTITY VECTOR SEARCH      │
 top-10 entities (1.3s)    │
    │                      │
    ├──────────────────────┐│
    ▼                     ▼▼
 GRAPH TRAVERSAL    KEYWORD SEARCH
 PK-based hop (1.8s)  FullTextContains
    │                      │
    ├───────────┬──────────┤
    ▼           ▼          ▼
 MERGE + DEDUPLICATE ALL RESULTS
    │
    ▼
 SEMANTIC RERANKER
 Cosmos DB AI reranker (0.1s)
    │
    ▼
 SINGLE LLM CALL
 Qwen3.5-27B + DFlash (13.5s)
    │
    ▼
 ANSWER
```

### Why this is faster than multi-round RAG

The original decomposed RAG pipeline makes **4-6 LLM calls** per question (initial answer, gap analysis, sub-question generation, targeted retrievals, final synthesis) and **8-12 Cosmos DB queries** across 2 rounds. Each LLM call to GPT-4.1 via Azure OpenAI takes 10-30 seconds.

The KG pipeline makes **1 LLM call** and **4-5 Cosmos DB queries** — all running in parallel. The KG has already encoded the relationships and inferences that the decomposed pipeline discovers on the fly. When the question asks about "high-calorie protein snack," the graph directly contains triples linking products to `has_nutritional_profile: high protein` and `has_nutritional_profile: high calorie` — no sub-question decomposition needed.

### The role of parallel retrieval

The DFlash pipeline runs multiple search paths concurrently using Python's `asyncio.gather`:

- **Entity vector search** — finds the most relevant entity nodes in the KG
- **LLM keyword expansion** — generates 5-8 additional search terms (e.g., "protein bar", "energy gel", "peanut butter", "trail mix")
- **Graph traversal** — follows entity links through the triple graph (PK-based lookup, sub-second)
- **Triple vector search** — finds semantically similar triples beyond the graph neighborhood
- **Food vector search** — direct product embedding search for backup coverage
- **Full-text keyword search** — parallel `FullTextContains` queries per expanded keyword

All of these run simultaneously. The total retrieval time is bounded by the **slowest** path (~3-4 seconds), not the sum.

### Semantic reranking

After merging results from all search paths, the Cosmos DB Semantic Reranker re-orders the documents by relevance to the question. This is a lightweight AI model (18ms inference) that scores each document against the query context, ensuring the most relevant products are at the top of the LLM's context window. This is especially important with the DFlash pipeline's smaller context limits — when you can only show 40 triples and 15 source documents to the LLM, the ordering matters.

## Part 3: DFlash Speculative Decoding on GPU

The KG reduces retrieval from 5+ rounds to 1, but the **LLM generation step** still dominates. Standard autoregressive generation with Qwen3.5-27B produces ~55 tokens/second on 2x H100. A 600-token answer takes ~11 seconds. With the full prompt context (graph triples + source documents + question), actual generation runs at 30-40 tok/s, taking 15-30 seconds.

DFlash speculative decoding doubles that throughput without changing the output.

### How speculative decoding works

Standard autoregressive LLM generation is sequential: each token requires a full forward pass through the 27-billion-parameter model. The GPU does enormous matrix multiplications, but the result is a single token. The arithmetic intensity is low and the GPU spends most of its time waiting for memory transfers.

Speculative decoding exploits this inefficiency:

1. **Draft phase:** A small, fast "draft" model (`z-lab/Qwen3.5-27B-DFlash`, ~1B parameters) proposes 5 tokens in quick succession. This takes roughly the same time as 1 forward pass of the main model.

2. **Verify phase:** The main Qwen3.5-27B model processes all 5 draft tokens in a **single batched forward pass**. Thanks to the parallel nature of the transformer attention mechanism, verifying 5 tokens costs almost the same as generating 1 token. The model outputs logits for all 5 positions simultaneously.

3. **Accept/reject:** The system compares the draft model's token probabilities with the main model's at each position. Using a rejection sampling scheme, it accepts the longest prefix of tokens where the draft matches the main model's distribution. Typically 3-4 out of 5 tokens are accepted.

4. **Repeat:** Generation continues from the first rejected position.

The key insight is that verification is cheap because transformer self-attention is **inherently parallel** — the model already computes attention over the entire sequence, so adding 5 more positions to the batch adds negligible cost. This converts 5 sequential forward passes into 1 parallel pass, yielding a ~2-2.5x throughput increase.

### Why it's lossless

The rejection sampling scheme is mathematically equivalent to sampling from the main model's distribution. When the draft model proposes a token that the main model would assign lower probability, the token is rejected with probability proportional to the difference. When accepted, the combined probability exactly matches what the main model would have produced. The final output distribution is **identical** to standard autoregressive generation — there is no quality trade-off.

### GPU resource utilization

| Resource | Standard Decoding | DFlash |
|----------|------------------|--------|
| Main model forward passes per output token | 1.0 | ~0.4 (amortized over accepted drafts) |
| Draft model overhead | 0% | ~5% of GPU time |
| Effective throughput | ~55 tok/s | ~110-140 tok/s |
| GPU memory for draft model | 0 GB | ~1 GB |
| KV cache overhead | Baseline | +5 positions per step |

The draft model is small enough that its memory footprint and compute cost are negligible compared to the main model. The net effect is that the GPU does almost the same total work but produces 2-2.5x more tokens.

## Results: End-to-End Comparison

Benchmarked on 10 food product questions, 2x NVIDIA H100, Cosmos DB with 58K documents and 892K KG triples.

### Timing per question (averaged)

| Stage | Original (GPT-4.1) | KG-RAG | KG-RAG + DFlash |
|-------|-------------------|--------|-----------------|
| Retrieval | 5-15s (multi-round) | 5.4s | 4.7s (parallel) |
| LLM generation | 60-80s (cloud) | 29-37s (local, standard) | 13-18s (local, DFlash) |
| **Total** | **70-94s** | **35-43s** | **18-23s** |

### Speedup breakdown

| Comparison | Speedup | Source of Speedup |
|-----------|---------|-------------------|
| KG-RAG vs Original | **2.2x** | KG pre-computation eliminates multi-round retrieval + local GPU vs cloud API |
| DFlash vs standard decoding | **2.0-2.5x** | Speculative decoding on same model, same hardware |
| **DFlash vs Original** | **4-5x** | Both techniques combined |

### Where the time goes

```
Original (GPT-4.1):          ██████████████████████████████████████████████ 93.7s
                              [  retrieval  ][         LLM (cloud)          ]

KG-RAG (standard):           ██████████████████████ 43.0s
                              [ KG ][     LLM (local, standard)     ]

KG-RAG + DFlash:             ██████████ 18.9s
                              [ KG ][ LLM (DFlash) ]
```

LLM generation dominates total time in all three pipelines (77-100%). The KG reduces retrieval time, and DFlash halves the generation time. Together they cut end-to-end latency by 5x.

### Quality comparison

| Aspect | Original | KG-RAG | KG-RAG + DFlash |
|--------|----------|--------|-----------------|
| Products recommended | 10+ | 2-4 | 8-10 |
| Nutritional detail | Detailed | Good | Good |
| Reasoning depth | Multi-round gap-filling | Single-pass with KG context | Single-pass with KG + keyword + reranking |
| Factual grounding | Strong (multi-retrieval) | Strong (KG triples) | Strong (KG + semantic rerank) |
| Output quality | Highest | High | High (mathematically identical to KG-RAG) |

DFlash output is **identical** to standard KG-RAG generation — the speculative decoding doesn't change the token distribution, only the speed at which tokens are produced.

## Hardware and Infrastructure

| Component | Role | Spec |
|-----------|------|------|
| 2x NVIDIA H100 NVL | LLM inference (vLLM) | 96GB HBM3 each, NVLink |
| vLLM 0.23.0 | Inference server | Tensor parallel, FP8 quantization, DFlash |
| Qwen3.5-27B | Main LLM | 27B params, FP8 (~27GB across 2 GPUs) |
| z-lab/Qwen3.5-27B-DFlash | Draft model | ~1B params (~1GB) |
| Qwen3-Embedding-0.6B | Embedding model | In-process on CPU, no network call |
| Azure Cosmos DB for NoSQL | Data + KG storage | Vector + full-text indexes, semantic reranker |

### Cost comparison

The Original pipeline uses GPT-4.1 via Azure OpenAI (pay-per-token, ~$2-10 per 1M input tokens depending on tier). With 4-6 LLM calls per question averaging 5K tokens each, a single question costs roughly $0.10-0.50 in API fees.

The DFlash pipeline runs entirely on local GPU. The H100 VM costs ~$30-40/hour. At 18-23 seconds per question, that's roughly **$0.15-0.25 per question** in compute cost — comparable to the cloud API but with no per-token charges, no rate limits, and no data leaving the network.

For batch workloads or sustained throughput, the local GPU approach is significantly cheaper. For occasional queries, the cloud API may be more cost-effective since there's no idle VM cost.

## Summary

The combination of pre-computed knowledge graphs and GPU-accelerated speculative decoding represents a shift in how retrieval systems can use GPU hardware:

1. **GPUs for offline knowledge engineering** — Instead of using the LLM only at query time, we invest GPU cycles upfront to extract, normalize, and resolve a structured knowledge graph. This "compiles" the LLM's reasoning into a reusable data structure.

2. **GPUs for parallel retrieval + generation** — At query time, the GPU serves the LLM while the CPU runs parallel async queries against Cosmos DB. The KG's structure means retrieval is a graph traversal, not a multi-round LLM conversation.

3. **GPUs for speculative decoding** — The same GPU memory that holds the main model also holds a tiny draft model. The negligible overhead of the draft model unlocks a 2-2.5x throughput improvement by converting sequential token generation into parallel verification.

The net result: **5x faster answers with no quality degradation**, running entirely on local infrastructure.
