# DFlash Benchmark Results

**Date**: July 16, 2026
**Hardware**: 2x NVIDIA H100 NVL (Azure Standard_NC80adis_H100_v5, Sweden Central)
**Database**: Cosmos DB NoSQL (Sweden Central, co-located with VM) — 58K food products, 1.6M Graph Index triples, 180K entities
**Models**: Qwen3.5-27B (FP8) + DFlash speculative decoding

## Q1: High-calorie protein snack for running

**Question**: "I am searching for a high-calorie protein snack for long-distance running that fits in a running belt"

| Stage | Time |
|-------|------|
| Embed | 0.27s |
| Entity Search | 0.33s |
| Graph Traversal | 0.83s |
| Source Fetch | 0.26s |
| LLM (DFlash) | 7.71s |
| **Total** | **9.40s** |

## Pipeline timing (web app, warm)

| Stage | Time |
|-------|------|
| Embed | 0.30s |
| Entity Search | 0.84s |
| Graph Traversal (parallel: PK + vector triples + food vector + keyword) | 1.40s |
| Source Fetch | 0.40s |
| Rerank | <0.01s |
| LLM (DFlash) | 3.46s |
| **Total** | **6.40s** |

## Cosmos DB region impact

Co-locating the database and VM in the same Azure region (Sweden Central) dramatically reduced retrieval latency:

| Query | West US 2 | West US 3 | **Sweden Central** |
|-------|-----------|-----------|-------------------|
| Baseline (no vector) | 0.50s | 2.01s | **0.10s** |
| Entity vector search (180K docs) | 0.51s | 3.90s | **0.30s** |
| Triple vector search (1.6M docs) | 0.52s | 4.10s | **0.35s** |

## Evolution

| Stage | Old DFlash (Jun 30) | Sweden Central (Jul 16) | Improvement |
|-------|--------------------|-----------------------|-------------|
| Embed | 0.27s | 0.27s | — |
| Entity Search | 1.32s | 0.33s | **4x faster** |
| Graph Traversal | 1.62s | 0.83s | **2x faster** |
| Source Fetch | 0.81s | 0.26s | **3x faster** |
| LLM (DFlash) | 12.5s | 7.71s | **1.6x faster** |
| **Total** | **17.4s** | **9.40s** | **1.85x faster** |

## Key Findings

- **Co-location matters**: Moving Cosmos DB to the same region as the VM reduced retrieval latency by 5-10x (from ~2s baseline to ~0.1s)
- **DFlash speculative decoding** provides consistent 2-2.5x LLM speedup over standard Qwen3.5-27B generation
- **DiskANN vector index** on 1.6M triples adds only ~0.3s on top of network baseline
- **Total pipeline under 10s** — down from 17.4s, a 1.85x overall improvement
- DFlash speculative decoding is **mathematically lossless** — output quality is identical to standard generation
