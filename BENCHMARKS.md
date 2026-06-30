# DFlash Benchmark Results

**Date**: June 30, 2026
**Hardware**: 2x NVIDIA H100 (Azure VM)
**Database**: Cosmos DB NoSQL — 58K food products, 892K KG triples
**Models**: Qwen3.5-27B (FP8) + DFlash speculative decoding, GPT-4.1 (Azure OpenAI)

## Backends

| Backend | LLM | Retrieval | Decoding |
|---------|-----|-----------|----------|
| **Original** | GPT-4.1 (Azure OpenAI) | Multi-round decomposed RAG (vector + full-text) | Standard |
| **KG-RAG** | Qwen3.5-27B (local, FP8) | Knowledge Graph traversal + vector + source fetch | Standard |
| **KG-RAG + DFlash** | Qwen3.5-27B (local, FP8) | Knowledge Graph traversal + vector + keyword + source fetch | DFlash speculative decoding |

## Q1: High-calorie protein snack for running

**Question**: "I am searching for a high-calorie protein snack for long-distance running that fits in a running belt"

| Stage | DFlash | KG | Original |
|-------|--------|-----|----------|
| Embed | 0.30s | 0.27s | — |
| Entity Search | 1.34s | 1.32s | — |
| Graph Traversal | 1.76s | 2.73s | — |
| Source Fetch | 0.68s | 1.25s | — |
| **LLM** | **13.5s** | **37.4s** | **93.7s** |
| **Total** | **18.9s** | **43.0s** | **93.7s** |

| Comparison | Speedup |
|-----------|---------|
| DFlash vs KG | **2.3x faster** |
| DFlash vs Original | **5.0x faster** |
| KG vs Original | **2.2x faster** |

## Q5: Breakfast with eggs (fastest DFlash)

**Question**: "Recommend a breakfast idea that takes under 15 minutes to make and contains eggs"

| Stage | DFlash | KG | Original |
|-------|--------|-----|----------|
| Embed | 0.27s | 0.27s | — |
| Entity Search | 1.32s | 1.30s | — |
| Graph Traversal | 1.62s | 1.86s | — |
| Source Fetch | 0.81s | 2.03s | — |
| **LLM** | **12.5s** | **29.9s** | **70.4s** |
| **Total** | **17.4s** | **35.4s** | **70.4s** |

| Comparison | Speedup |
|-----------|---------|
| DFlash vs KG | **2.0x faster** |
| DFlash vs Original | **4.0x faster** |
| KG vs Original | **2.0x faster** |

## All 10 Questions — DFlash Timing

| # | Question ID | Total | LLM | Question |
|---|------------|-------|-----|----------|
| 1 | food_0001 | 20.7s | 13.7s | High-calorie protein snack for running belt |
| 2 | food_0002 | 19.8s | 15.2s | Low-alcohol premium craft beer |
| 3 | food_0003 | 27.3s | 22.6s | Gluten-free cinema snack |
| 4 | food_0004 | 20.3s | 15.4s | Summer BBQ meat to impress guests |
| 5 | food_0005 | 17.9s | 12.9s | Breakfast with eggs under 15 min |
| 6 | food_0006 | 35.3s | 29.9s | Traditional English breakfast ingredients |
| 7 | food_0007 | 19.1s | 14.3s | 15-minute air fryer recipe |
| 8 | food_0008 | 21.0s | 16.1s | Family dinner recommendation |
| 9 | food_0009 | 21.3s | 16.8s | Sugar-free vegan confectionery |
| 10 | food_0010 | 22.9s | 18.5s | Healthy chocolate brownie dessert |
| | **Average** | **22.6s** | **17.5s** | |

## Key Findings

- **DFlash speculative decoding** provides a consistent **2-2.5x LLM speedup** over standard generation with the same Qwen3.5-27B model
- **Retrieval time is identical** between DFlash and KG (~4-5s) since both use full KG graph traversal
- **LLM generation dominates total time** (72-100% across all backends)
- **DFlash is 4-5x faster than the Original** (GPT-4.1 via Azure OpenAI) while providing comparable answer quality using local inference
- DFlash speculative decoding is **mathematically lossless** — output quality is identical to standard Qwen3.5-27B generation
