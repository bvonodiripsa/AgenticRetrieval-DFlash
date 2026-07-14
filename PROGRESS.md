# Progress — Read First Tomorrow

**Last updated**: Jul 14, 2026 (Monday night)

## What was done tonight

### Cosmos DB Migration: divdet → divdet-provisioned

Migrated all data from the serverless `divdet` account to a new provisioned-throughput `divdet-provisioned` account to eliminate throttling:

| Container | Documents | Time | Rate |
|-----------|-----------|------|------|
| `food` | 58,432 | 61 min | ~16/s (limited by serverless source) |
| `triples` | 1,598,098 | ~101 min | 262/s |
| `entities` | 181,082 | 11.5 min | 259/s |
| **Total** | **1,837,612** | **~113 min** | **262/s** |

**Key trick**: Temporarily disabled indexing (set `indexingMode: none`) during bulk upload, which dropped per-upsert RU cost from ~50 to ~5. After upload, re-enabled consistent indexing + DiskANN vector indexes.

### Schema change: 1-letter field names

The new `triples` and `entities` containers use abbreviated field names to save storage:

**Triples**: `s` (subject), `p` (predicate), `o` (object), `f` (confidence), `n` (confirmations), `d` (source_chunks), `e` (embedding)

**Entities**: `n` (name), `t` (description), `r` (relation_count), `d` (source_chunks), `e` (embedding)

Partition key for triples is `/s` (was `/pk`).

### Config & code updates

All config files (`config_kg_dflash.yaml`, `config_kg_oldqwen.yaml`, `config_original_local.yaml`, `config_kg.yaml`, `config_local.yaml`, `config_solar.yaml`, `config_ray_build.yaml`) updated to point to `divdet-provisioned`.

All SQL queries in `kg_query.py` and `api.py` updated to use the new 1-letter field names with `AS` aliases for backward compatibility.

## What to test tomorrow

1. **Start vLLM** (if not running):
   ```bash
   cd /home/azureuser/AgenticRetrieval-DFlash
   source .venv/bin/activate
   CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.5-27B \
     --tensor-parallel-size 2 --max-model-len 16384 --max-num-batched-tokens 16384 \
     --gpu-memory-utilization 0.92 --dtype float16 --quantization fp8 \
     --spec-model z-lab/Qwen3.5-27B-DFlash --spec-tokens 5 \
     --enable-prefix-caching --port 8000
   ```

2. **Start web app**:
   ```bash
   AZURE_COSMOS_SEMANTIC_RERANKER_INFERENCE_ENDPOINT="https://divdet.westus3.dbinference.azure.com" \
     python -m uvicorn api:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 120
   ```

3. **Test all three backends** (Original, Graph Index, DFlash) from the web UI at `http://<public-ip>:8080`

4. **Verify**: Graph Index and DFlash return good results with the new provisioned account

## RBAC needed for divdet-provisioned

The new account needs the same RBAC roles as `divdet`. Ask admin to assign:

```bash
# Data access
az cosmosdb sql role assignment create \
  --account-name divdet-provisioned \
  --resource-group ams-cosmos-db \
  --role-definition-name "Cosmos DB Built-in Data Contributor" \
  --principal-id "<your-user-object-id>" \
  --scope "/"

# Semantic reranker (if using)
az role assignment create \
  --role "Semantic Reranker User" \
  --assignee-object-id "<your-user-object-id>" \
  --assignee-principal-type "User" \
  --scope "/subscriptions/b7d41fc8-d35d-41db-92ed-1f7f1d32d4d9/resourceGroups/ams-cosmos-db/providers/Microsoft.InferenceService/inferenceAccounts/divdet-provisioned"
```

## Remaining items

- Scale down entities container throughput to 10K (locked from prior scale operation, will auto-resolve)
- Test semantic reranker with new account (may need new RBAC assignment on `divdet-provisioned`)
- Confirm all three web app backends work end-to-end with provisioned account
