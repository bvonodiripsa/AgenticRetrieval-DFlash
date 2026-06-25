"""Re-embed all food documents using in-process Qwen3-Embedding-0.6B on GPU, upload to Cosmos DB."""
import asyncio
import json
import torch
import yaml
import time
import sys
from transformers import AutoModel, AutoTokenizer
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024
UPLOAD_BATCH = 20

print(f"Loading {MODEL_ID} on GPU...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32)
model = model.cuda(0)  # Use GPU 1 (GPU 0 is used more by vLLM)
model.eval()
print("Model loaded on GPU.", flush=True)

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using mean pooling + L2 normalize on GPU."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.cuda(0) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        token_embs = outputs.last_hidden_state.float() * attention_mask
        emb = token_embs.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    vecs = emb.cpu().tolist()
    return [v[:EMBED_DIM] for v in vecs]

def generate_embedding_text(doc: dict, text_fields: list[str]) -> str:
    parts = []
    for field in text_fields:
        value = doc.get(field)
        if value is None or value == "":
            continue
        label = field.replace("_", " ").title()
        if isinstance(value, list):
            parts.append(f"{label}: {', '.join(str(v) for v in value)}")
        else:
            parts.append(f"{label}: {value}")
    return "\n".join(parts) if parts else json.dumps(doc, ensure_ascii=False)

async def main():
    with open("config_local.yaml") as f:
        cfg = yaml.safe_load(f)

    cosmos_cfg = cfg["cosmos"]
    source = cosmos_cfg["sources"][0]
    text_fields = source["embedding_text_fields"]
    container_name = source["container_name"]
    db_name = cosmos_cfg["database_name"]

    # Read docs from local JSONL (much faster than Cosmos DB full scan)
    jsonl_path = source["documents_root"]
    print(f"Reading docs from {jsonl_path}...", flush=True)
    docs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    print(f"Total docs: {len(docs)}", flush=True)

    # Embed all docs in batches on GPU
    print(f"Embedding {len(docs)} docs on GPU...", flush=True)
    embed_start = time.time()
    EMBED_BATCH = 128
    all_embeddings = []
    for i in range(0, len(docs), EMBED_BATCH):
        batch = docs[i:i+EMBED_BATCH]
        texts = [generate_embedding_text(d, text_fields) for d in batch]
        embs = embed_batch(texts)
        all_embeddings.extend(embs)
        if (i // EMBED_BATCH) % 10 == 0:
            print(f"  Embedded {len(all_embeddings)}/{len(docs)}", flush=True)
    embed_elapsed = time.time() - embed_start
    print(f"Embedding done in {embed_elapsed:.1f}s ({len(docs)/embed_elapsed:.0f} docs/s)", flush=True)

    # Upload to Cosmos DB
    print(f"Uploading to Cosmos DB {db_name}/{container_name}...", flush=True)
    credential = AzureCliCredential(tenant_id=cosmos_cfg["tenant_id"])
    client = CosmosClient(cosmos_cfg["uri"], credential=credential)
    db = client.get_database_client(db_name)
    container = db.get_container_client(container_name)

    upload_start = time.time()
    updated = 0
    errors = 0
    sem = asyncio.Semaphore(5)

    async def upsert_one(doc, emb):
        nonlocal updated, errors
        doc["e"] = emb
        async with sem:
            try:
                await container.upsert_item(doc)
                updated += 1
            except Exception as ex:
                errors += 1
                if errors <= 5:
                    print(f"  Error: {ex}", flush=True)

    for i in range(0, len(docs), UPLOAD_BATCH):
        batch_docs = docs[i:i+UPLOAD_BATCH]
        batch_embs = all_embeddings[i:i+UPLOAD_BATCH]
        await asyncio.gather(*[upsert_one(d, e) for d, e in zip(batch_docs, batch_embs)])
        if (i // UPLOAD_BATCH) % 50 == 0:
            elapsed = time.time() - upload_start
            rate = updated / elapsed if elapsed > 0 else 0
            print(f"  Uploaded {updated}/{len(docs)} ({rate:.0f} docs/s), errors={errors}", flush=True)

    elapsed = time.time() - upload_start
    print(f"\nDone! Updated {updated}/{len(docs)} in {elapsed:.1f}s, errors={errors}", flush=True)

    await credential.close()
    await client.close()

asyncio.run(main())
