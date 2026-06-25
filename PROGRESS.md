# AgenticRetrieval-DFlash Progress

## Status: Ready to test Qwen3.5-27B + DFlash speculative decoding

## What's Done
1. **KG Builder** (`kg_builder.py`) — fully working, tested, has checkpoint/resume and `--time-limit`
2. **KG Query Engine** (`kg_query.py`) — working with parallelized queries + vector augmentation
3. **Improved Prompts** (`prompts_kg_food.py`) — now extracts semantic triples:
   - `suitable_for_occasion` (breakfast, snack, BBQ, cinema, dessert, etc.)
   - `has_cooking_method` (grill, air fryer, microwave, ready to eat, etc.)
   - `has_convenience` (instant, under 5 min, under 15 min)
   - `has_nutritional_profile` (high protein, high calorie, low sugar, etc.)
   - `has_portability` (pocket sized, single serving, family size)
   - `suitable_for_audience` (athletes, families, health conscious)
4. **Config** (`config_kg.yaml`) — query config has `vector_augment_k: 15` for hybrid retrieval
5. **Containers** — `kg_triples_food` (empty, ready), `kg_entities_food` (empty, ready)

## What's Needed Next (Tomorrow)

### Step 1: Clear checkpoint and run question-driven build (~8 min)
```bash
cd /home/azureuser/AgenticRetrieval-KG
source .venv/bin/activate  # symlinked to /home/azureuser/AgenticRetrieval/.venv
rm -f out_kg/checkpoint_raw_triples.json
python -u kg_builder.py --config config_kg.yaml --question-driven --question-index all --question-k 30 --extraction-rounds 1 --concurrency 20
```
This processes ~160 docs relevant to 10 questions. Ensures KG covers all benchmark questions.

### Step 2: Run full 58K corpus build (~25h, use --time-limit for partial)
```bash
rm -f out_kg/checkpoint_raw_triples.json  # fresh start for full build
python -u kg_builder.py --config config_kg.yaml --extraction-rounds 1 --concurrency 20 --time-limit 7200
```
- Processes all docs in `food` container (58K)
- Stops after 2 hours (7200s), saves checkpoint
- Resume next day: just re-run same command (auto-detects checkpoint)
- Estimated: ~4500 docs in 2 hours at current rate

### Step 3: Run benchmark
```bash
python -u kg_query.py --config config_kg.yaml --questions data/food.json
```
Expected: ~12-15s wall time for 10 questions (parallel), better quality with semantic triples.

## Key Files
- `/home/azureuser/AgenticRetrieval-KG/kg_builder.py` — build pipeline
- `/home/azureuser/AgenticRetrieval-KG/kg_query.py` — query engine + benchmark
- `/home/azureuser/AgenticRetrieval-KG/prompts_kg_food.py` — extraction + answer prompts
- `/home/azureuser/AgenticRetrieval-KG/config_kg.yaml` — all config
- `/home/azureuser/AgenticRetrieval-KG/data/food.json` — 10 benchmark questions
- `/home/azureuser/AgenticRetrieval-KG/out_kg/` — output dir for results + checkpoints

## Infrastructure
- **vLLM**: Qwen3.5-27B with DFlash speculative decoding (TP=2, port 8000)
- **DFlash draft model**: `z-lab/Qwen3.5-27B-DFlash` (block diffusion drafter)
- **Cosmos DB**: `divdet` account, `food` database, RBAC auth (tenant 43083d15...)
- **Embedding**: in-process Qwen3-Embedding-0.6B on CPU (no external service needed)
- **Python venv**: `/home/azureuser/AgenticRetrieval/.venv` (symlinked into this repo)

## DFlash Migration Notes
- Migrated from Qwen2.5-32B-Instruct to **Qwen3.5-27B** (5B smaller, better quality)
- DFlash provides **3-4x lossless speedup** via block diffusion speculative decoding
- Requires vLLM >= 0.20.1 + flash-attn backend
- FP8 quantization replaced with bfloat16 (DFlash requires bfloat16 KV cache)
- Qwen3.5 thinking tokens (`<think>...</think>`) stripped in both API and Ray modes

### vLLM Server Launch (API mode)
```bash
vllm serve Qwen/Qwen3.5-27B \
  --tensor-parallel-size 2 \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-model-len 16384 \
  --max-num-batched-tokens 32768 \
  --dtype bfloat16
```

### Test run (validate quality on small batch)
```bash
python test_dflash.py --config config_kg.yaml --num-docs 20
```

## Previous Benchmark Results
| Approach | Wall Time | Quality |
|----------|-----------|---------|
| Azure GPT-4.1 baseline | 91s | Good (reference) |
| AgenticRetrieval local vLLM | 78s | Good (2 rounds) |
| KG-RAG v1 (old prompts, 160 docs) | 11.6s | Weaker on creative/recipe Qs |
| KG-RAG v2 (new prompts + vector augment) | TBD | Expected improvement |

## GitHub
- Repo: https://github.com/bvonodiripsa/AgenticRetrieval-KG
- Auth: `git remote set-url origin https://bvonodiripsa:<PAT>@github.com/bvonodiripsa/AgenticRetrieval-KG.git`
