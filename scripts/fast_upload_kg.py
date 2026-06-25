#!/usr/bin/env python
"""Fast parallel upload of KG to Cosmos DB with retry + backoff for 429s."""
import asyncio
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.identity.aio import AzureCliCredential

COSMOS_URI = "https://divdet.documents.azure.com:443/"
TENANT_ID = "43083d15-7273-40c1-b7db-39efd9ccc17a"
DB_NAME = "food"
TRIPLES_CONTAINER = "kg_triples_food"
ENTITIES_CONTAINER = "kg_entities_food"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CONCURRENCY = 30
MAX_RETRIES = 8
BASE_BACKOFF = 1.0


def get_embedding_model():
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Using device: {device}")
    return SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)


async def upload_triples(triples: list[dict], embed_model):
    cred = AzureCliCredential(tenant_id=TENANT_ID)
    cosmos = CosmosClient(COSMOS_URI, credential=cred)
    db = cosmos.get_database_client(DB_NAME)
    tc = db.get_container_client(TRIPLES_CONTAINER)

    sem = asyncio.Semaphore(CONCURRENCY)
    uploaded = 0
    errors = 0

    import numpy as np
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out_kg")
    triple_cache = os.path.join(cache_dir, "triples_embeddings.npy")

    if os.path.exists(triple_cache):
        print(f"  [STAGE] Loading cached triple embeddings from {triple_cache}...")
        sys.stdout.flush()
        t0 = time.time()
        all_embs = np.load(triple_cache).tolist()
        print(f"    Loaded {len(all_embs):,} embeddings ({time.time()-t0:.0f}s)")
        sys.stdout.flush()
    else:
        print(f"  [STAGE] Embedding {len(triples):,} triples (GPU)...")
        sys.stdout.flush()
        descs = [f"{t['subject']} {t['predicate']} {t['object']}" for t in triples]
        BATCH = 4096
        all_embs_np = []
        t0 = time.time()
        for i in range(0, len(descs), BATCH):
            embs = embed_model.encode(descs[i:i+BATCH], normalize_embeddings=True)
            all_embs_np.append(embs)
            if (i + BATCH) % 40000 < BATCH:
                print(f"    {min(i+BATCH, len(descs)):,}/{len(descs):,} embedded")
                sys.stdout.flush()
        all_embs_arr = np.vstack(all_embs_np)
        np.save(triple_cache, all_embs_arr)
        print(f"    Embedding done ({time.time()-t0:.0f}s), cached to {triple_cache}")
        sys.stdout.flush()
        all_embs = all_embs_arr.tolist()

    # Upload with high concurrency
    print(f"  [STAGE] Uploading {len(triples):,} triples (concurrency={CONCURRENCY})...")
    sys.stdout.flush()
    t0 = time.time()

    async def _upsert(i):
        nonlocal uploaded, errors
        t = triples[i]
        content_key = f"{t['subject']}|{t['predicate']}|{t['object']}".lower()
        doc_id = "t_" + hashlib.md5(content_key.encode()).hexdigest()[:12]
        doc = {
            "id": doc_id,
            "pk": t["subject"].lower()[:100],
            "subject": t["subject"],
            "predicate": t["predicate"],
            "object": t["object"],
            "confidence": t.get("confidence", 0.8),
            "confirmations": t.get("confirmations", 1),
            "source_chunks": t.get("source_chunks", [])[:10],
            "embedding": all_embs[i],
        }
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    await tc.upsert_item(doc)
                    uploaded += 1
                    return
                except CosmosHttpResponseError as e:
                    if e.status_code == 429:
                        retry_after = float(e.headers.get("x-ms-retry-after-ms", 0)) / 1000
                        wait = max(retry_after, BASE_BACKOFF * (2 ** attempt)) + random.uniform(0, 0.5)
                        await asyncio.sleep(wait)
                    else:
                        errors += 1
                        if errors <= 10:
                            print(f"    Error (status {e.status_code}): {e.message[:120]}")
                        return
                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        print(f"    Error: {e}")
                    return
            errors += 1

    CHUNK = 2000
    for start in range(0, len(triples), CHUNK):
        end = min(start + CHUNK, len(triples))
        tasks = [_upsert(i) for i in range(start, end)]
        await asyncio.gather(*tasks)
        elapsed = time.time() - t0
        rate = uploaded / max(elapsed, 1)
        eta = (len(triples) - uploaded) / max(rate, 1)
        print(f"    {uploaded:,}/{len(triples):,} uploaded ({rate:.0f}/s, "
              f"errors={errors}, ETA {eta:.0f}s)")
        sys.stdout.flush()

    print(f"    Done: {uploaded:,} triples in {time.time()-t0:.0f}s ({errors} errors)")
    await cosmos.close()


async def upload_entities(triples: list[dict], embed_model):
    print(f"\n  [STAGE] Building entity index...")
    ent_map: dict[str, dict] = {}
    for t in triples:
        for role in ("subject", "object"):
            name = t[role]
            if name not in ent_map:
                ent_map[name] = {"name": name, "relations": [], "source_chunks": set()}
            rel = f"{t['predicate']} -> {t['object']}" if role == "subject" else f"{t['subject']} -> {t['predicate']}"
            if len(ent_map[name]["relations"]) < 30:
                ent_map[name]["relations"].append(rel)
            for sc in t.get("source_chunks", [])[:3]:
                ent_map[name]["source_chunks"].add(sc)

    ent_list = list(ent_map.values())
    print(f"    {len(ent_list):,} unique entities")

    import numpy as np
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out_kg")
    entity_cache = os.path.join(cache_dir, "entities_embeddings.npy")

    ent_descs = [f"{e['name']}. Relations: {'; '.join(e['relations'][:15])}" for e in ent_list]

    if os.path.exists(entity_cache):
        print(f"  [STAGE] Loading cached entity embeddings...")
        sys.stdout.flush()
        t0 = time.time()
        ent_embs = np.load(entity_cache).tolist()
        print(f"    Loaded {len(ent_embs):,} embeddings ({time.time()-t0:.0f}s)")
        sys.stdout.flush()
    else:
        print(f"  [STAGE] Embedding {len(ent_list):,} entities (GPU)...")
        sys.stdout.flush()
        t0 = time.time()
        BATCH = 4096
        ent_embs_np = []
        for i in range(0, len(ent_descs), BATCH):
            embs = embed_model.encode(ent_descs[i:i+BATCH], normalize_embeddings=True)
            ent_embs_np.append(embs)
            if (i + BATCH) % 40000 < BATCH:
                print(f"    {min(i+BATCH, len(ent_descs)):,}/{len(ent_descs):,} embedded")
                sys.stdout.flush()
        ent_embs_arr = np.vstack(ent_embs_np)
        np.save(entity_cache, ent_embs_arr)
        print(f"    Embedding done ({time.time()-t0:.0f}s), cached to {entity_cache}")
        sys.stdout.flush()
        ent_embs = ent_embs_arr.tolist()

    # Upload
    cred = AzureCliCredential(tenant_id=TENANT_ID)
    cosmos = CosmosClient(COSMOS_URI, credential=cred)
    db = cosmos.get_database_client(DB_NAME)
    ec = db.get_container_client(ENTITIES_CONTAINER)

    sem = asyncio.Semaphore(CONCURRENCY)
    uploaded = 0
    errors = 0
    t0 = time.time()

    print(f"  [STAGE] Uploading {len(ent_list):,} entities...")
    sys.stdout.flush()

    async def _upsert(i):
        nonlocal uploaded, errors
        e = ent_list[i]
        doc_id = "e_" + hashlib.md5(e["name"].lower().encode()).hexdigest()[:12]
        doc = {
            "id": doc_id,
            "pk": e["name"].lower()[:100],
            "name": e["name"],
            "description": ent_descs[i][:1000],
            "relation_count": len(e["relations"]),
            "source_chunks": list(e["source_chunks"])[:50],
            "embedding": ent_embs[i],
        }
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    await ec.upsert_item(doc)
                    uploaded += 1
                    return
                except CosmosHttpResponseError as e_err:
                    if e_err.status_code == 429:
                        retry_after = float(e_err.headers.get("x-ms-retry-after-ms", 0)) / 1000
                        wait = max(retry_after, BASE_BACKOFF * (2 ** attempt)) + random.uniform(0, 0.5)
                        await asyncio.sleep(wait)
                    else:
                        errors += 1
                        if errors <= 10:
                            print(f"    Error (status {e_err.status_code}): {e_err.message[:120]}")
                        return
                except Exception as e_err:
                    errors += 1
                    if errors <= 10:
                        print(f"    Error: {e_err}")
                    return
            errors += 1

    CHUNK = 2000
    for start in range(0, len(ent_list), CHUNK):
        end = min(start + CHUNK, len(ent_list))
        tasks = [_upsert(i) for i in range(start, end)]
        await asyncio.gather(*tasks)
        elapsed = time.time() - t0
        rate = uploaded / max(elapsed, 1)
        eta = (len(ent_list) - uploaded) / max(rate, 1)
        print(f"    {uploaded:,}/{len(ent_list):,} uploaded ({rate:.0f}/s, ETA {eta:.0f}s)")
        sys.stdout.flush()

    print(f"    Done: {uploaded:,} entities in {time.time()-t0:.0f}s ({errors} errors)")
    await cosmos.close()


async def main():
    print("=" * 60)
    print("  Fast KG Upload to Cosmos DB")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    triples_file = os.path.join(script_dir, "out_kg", "triples_deduped.json")
    print(f"\n  [STAGE] Loading triples...")
    sys.stdout.flush()
    with open(triples_file) as f:
        triples = json.load(f)
    print(f"    {len(triples):,} triples loaded")
    sys.stdout.flush()

    print(f"\n  [STAGE] Loading embedding model...")
    sys.stdout.flush()
    embed_model = get_embedding_model()
    print(f"    Model ready")
    sys.stdout.flush()

    await upload_triples(triples, embed_model)
    await upload_entities(triples, embed_model)

    print(f"\n{'='*60}")
    print(f"  [COMPLETE] KG uploaded to Cosmos DB")
    print(f"{'='*60}")

    if os.environ.get("AUTO_DEALLOCATE"):
        print("\n  [AUTO] Deallocating VM in 60s...")
        sys.stdout.flush()
        import subprocess
        subprocess.run(
            ["az", "vm", "deallocate", "-n", "ams-agentic-h100",
             "-g", "AMS-COSMOSDB-LLM-RG", "--no-wait"],
            check=False,
        )


if __name__ == "__main__":
    asyncio.run(main())
