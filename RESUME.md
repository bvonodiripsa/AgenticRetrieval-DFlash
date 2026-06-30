# Resume Point — June 29, 2026

## What's Running
- **Web app**: `http://20.240.128.47:8080` — 3 backends (Original, KG-RAG, KG-RAG+DFlash) on food database (58K products)
- **vLLM server**: Qwen3.5-27B + DFlash (FP8, tensor-parallel 2) on port 8000
- **GitHub**: Latest code pushed to `bvonodiripsa/AgenticRetrieval-DFlash` (commit `d91f5da`)

## What Was Fixed Today
1. **Hybrid search was completely broken** — `ORDER BY RANK RRF(FullTextScore + VectorDistance)` returned `BadRequest` silently. DFlash was running on vector-only search this entire time.
2. **Replaced with individual FullTextContains keyword searches** — extract meaningful words from the question, run separate parallel Cosmos DB full-text queries per keyword, merge with vector results.
3. **Added LLM-based keyword expansion** — a lightweight LLM call generates 5-8 food-related search terms (e.g., "peanut", "energy bar") that aren't in the question but are semantically relevant. These run as additional parallel full-text searches.
4. **Increased product count** from 4-5 to 8-10, `max_answer_tokens` from 600 to 1200, `RERANKER_TOP_K` from 10 to 25.
5. **Result**: DFlash now finds products overlapping with Original (Reese's Sticks, Cream-Nut PB, Toffifee, etc.) — previously it missed all peanut-based products.

## Current Performance (Question 1)
| Backend | Time | Products |
|---------|------|----------|
| Original (GPT-4.1) | ~45s | 10+ detailed |
| KG-RAG | ~45s | 3-5 |
| DFlash | ~16s | 8 |

## Pending / Next Steps
1. **Cosmos DB Semantic Reranker** — code is integrated but blocked on RBAC permissions. Need an admin with Owner/User Access Administrator role to assign "Semantic Reranker User" to `aspiridonov@nvidia.com` on the `divdet` Cosmos account. Endpoint: `https://divdet.westus3.dbinference.azure.com`. Once enabled, it will rerank the merged vector+fulltext results for better quality.
2. **Speed optimization** — DFlash is at ~16s (retrieval 4s + LLM 10.7s + rerank 1.5s). The LLM keyword expansion adds ~1-2s. Could be optimized further.
3. **Quality gap** — Original still finds more products because it does multi-round decomposed RAG. The semantic reranker should help close this gap.

## Key Files
- `/home/azureuser/AgenticRetrieval-DFlash/api.py` — main API with all 3 backends
- `/home/azureuser/AgenticRetrieval-DFlash/config_kg_dflash.yaml` — DFlash config
- `/home/azureuser/AgenticRetrieval-DFlash/config_kg_oldqwen.yaml` — KG-RAG config
- `/home/azureuser/AgenticRetrieval-DFlash/config_original_local.yaml` — Original config (Azure OpenAI GPT-4.1)
- `/home/azureuser/AgenticRetrieval-DFlash/static/index.html` — frontend

## How to Restart

### vLLM (if needed)
```bash
cd /home/azureuser/AgenticRetrieval-DFlash && source .venv/bin/activate
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-27B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --quantization fp8 --dtype auto --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 --max-model-len 16384 --max-num-batched-tokens 16384 \
  --enable-prefix-caching --trust-remote-code --enforce-eager \
  --host 0.0.0.0 --port 8000 --api-key dummy
```

### Web app
```bash
cd /home/azureuser/AgenticRetrieval-DFlash && .venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8080 --log-level info
```
