#!/usr/bin/env python
"""Upload KG to Cosmos DB with throttling and retry. GPU-accelerated embeddings with .npy cache."""
import asyncio
import hashlib
import json
import os
import sys
import time

import numpy as np

COSMOS_URI = "https://divdet.documents.azure.com:443/"
COSMOS_KEY = os.environ["COSMOS_KEY"]
DB_NAME = "food"
TRIPLES_CONTAINER = "kg_triples_food"
ENTITIES_CONTAINER = "kg_entities_food"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CONCURRENCY = 150

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SCRIPT_DIR, "out_kg")
TRIPLES_FILE = os.path.join(OUT_DIR, "triples_deduped.json")
CHECKPOINT_FILE = os.path.join(OUT_DIR, "upload_checkpoint.json")
TRIPLES_EMB_CACHE = os.path.join(OUT_DIR, "triples_embeddings.npy")
ENTITIES_EMB_CACHE = os.path.join(OUT_DIR, "entities_embeddings.npy")


def get_embedding_model():
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")
    return SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)


def embed_texts(texts, embed_model, cache_path, label="items"):
    """Embed texts with GPU, cache to .npy. Returns list-of-lists."""
    if os.path.exists(cache_path):
        print(f"  [CACHE] Loading {label} embeddings from {os.path.basename(cache_path)}...")
        sys.stdout.flush()
        t0 = time.time()
        arr = np.load(cache_path)
        print(f"    Loaded {arr.shape[0]:,} embeddings ({time.time()-t0:.0f}s)")
        sys.stdout.flush()
        return arr.tolist()

    print(f"  [STAGE] Embedding {len(texts):,} {label}...")
    sys.stdout.flush()
    BATCH = 4096
    parts = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        embs = embed_model.encode(texts[i:i+BATCH], normalize_embeddings=True)
        parts.append(embs)
        if (i + BATCH) % 40000 < BATCH:
            print(f"    {min(i+BATCH, len(texts)):,}/{len(texts):,}")
            sys.stdout.flush()
    arr = np.vstack(parts)
    np.save(cache_path, arr)
    print(f"    Embedding done ({time.time()-t0:.0f}s), cached to {os.path.basename(cache_path)}")
    sys.stdout.flush()
    return arr.tolist()


async def upsert_with_retry(container, doc, sem, stats, max_retries=10):
    async with sem:
        for attempt in range(max_retries):
            try:
                await container.upsert_item(doc)
                return True
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "TooManyRequests" in err_str:
                    stats["throttled"] += 1
                    retry_ms = 0
                    if hasattr(e, "headers"):
                        retry_ms = int(e.headers.get("x-ms-retry-after-ms", 0))
                    wait = max(retry_ms / 1000, 0.1) + 0.05 * attempt
                    await asyncio.sleep(wait)
                elif "408" in err_str or "Timeout" in err_str:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    if attempt == max_retries - 1:
                        return False
                    await asyncio.sleep(0.2)
        return False


async def upload_triples(triples, all_embs, start_idx=0):
    from azure.cosmos.aio import CosmosClient
    cosmos = CosmosClient(COSMOS_URI, credential=COSMOS_KEY)
    db = cosmos.get_database_client(DB_NAME)
    tc = db.get_container_client(TRIPLES_CONTAINER)
    sem = asyncio.Semaphore(CONCURRENCY)
    stats = {"throttled": 0}

    uploaded = 0
    errors = 0
    t0 = time.time()
    total = len(triples)

    CHUNK = 5000
    for start in range(start_idx, total, CHUNK):
        end = min(start + CHUNK, total)
        tasks = []
        for i in range(start, end):
            t = triples[i]
            content_key = f"{t['subject']}|{t['predicate']}|{t['object']}".lower()
            doc_id = "t_" + hashlib.md5(content_key.encode()).hexdigest()[:12]
            doc = {
                "id": doc_id, "pk": t["subject"].lower()[:100],
                "subject": t["subject"], "predicate": t["predicate"], "object": t["object"],
                "confidence": t.get("confidence", 0.8),
                "confirmations": t.get("confirmations", 1),
                "source_chunks": t.get("source_chunks", [])[:10],
                "embedding": all_embs[i],
            }
            tasks.append(upsert_with_retry(tc, doc, sem, stats))

        results = await asyncio.gather(*tasks)
        uploaded += sum(1 for r in results if r)
        errors += sum(1 for r in results if not r)

        elapsed = time.time() - t0
        rate = uploaded / max(elapsed, 1)
        eta = (total - end) / max(rate, 1)
        print(f"  Triples: {end:,}/{total:,} ({uploaded:,} ok, {errors} err, "
              f"{rate:.0f}/s, {stats['throttled']} throttled, ETA {eta/3600:.1f}h)")
        sys.stdout.flush()

        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"triples_idx": end, "entities_done": False}, f)

    await cosmos.close()
    return uploaded, errors


async def upload_entities(triples, embed_model):
    from azure.cosmos.aio import CosmosClient

    print(f"\n  [STAGE] Building entity index from {len(triples):,} triples...")
    ent_map = {}
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
    sys.stdout.flush()

    ent_descs = [f"{e['name']}. Relations: {'; '.join(e['relations'][:15])}" for e in ent_list]
    ent_embs = embed_texts(ent_descs, embed_model, ENTITIES_EMB_CACHE, "entities")

    # Upload
    cosmos = CosmosClient(COSMOS_URI, credential=COSMOS_KEY)
    db = cosmos.get_database_client(DB_NAME)
    ec = db.get_container_client(ENTITIES_CONTAINER)
    sem = asyncio.Semaphore(CONCURRENCY)

    uploaded = 0
    errors = 0
    t0 = time.time()
    total = len(ent_list)

    print(f"  [STAGE] Uploading {total:,} entities...")
    sys.stdout.flush()
    stats = {"throttled": 0}

    CHUNK = 5000
    for start in range(0, total, CHUNK):
        end = min(start + CHUNK, total)
        tasks = []
        for i in range(start, end):
            e = ent_list[i]
            doc_id = "e_" + hashlib.md5(e["name"].lower().encode()).hexdigest()[:12]
            doc = {
                "id": doc_id, "pk": e["name"].lower()[:100],
                "name": e["name"], "description": ent_descs[i][:1000],
                "relation_count": len(e["relations"]),
                "source_chunks": list(e["source_chunks"])[:50],
                "embedding": ent_embs[i],
            }
            tasks.append(upsert_with_retry(ec, doc, sem, stats))

        results = await asyncio.gather(*tasks)
        uploaded += sum(1 for r in results if r)
        errors += sum(1 for r in results if not r)

        elapsed = time.time() - t0
        rate = uploaded / max(elapsed, 1)
        eta = (total - end) / max(rate, 1)
        print(f"  Entities: {end:,}/{total:,} ({uploaded:,} ok, {errors} err, "
              f"{rate:.0f}/s, {stats['throttled']} throttled, ETA {eta/60:.0f}m)")
        sys.stdout.flush()

    await cosmos.close()
    return uploaded, errors


async def main():
    print("=" * 60)
    print("  KG Upload to Cosmos DB (with retry)")
    print("=" * 60)
    sys.stdout.flush()

    # Load checkpoint
    start_idx = 0
    entities_done = False
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
        start_idx = ckpt.get("triples_idx", 0)
        entities_done = ckpt.get("entities_done", False)
        print(f"  Resuming from checkpoint: triples_idx={start_idx:,}")

    # Load triples
    print(f"\n  [STAGE] Loading triples...")
    sys.stdout.flush()
    with open(TRIPLES_FILE) as f:
        triples = json.load(f)
    print(f"    {len(triples):,} triples")

    # Load embedding model
    print(f"\n  [STAGE] Loading embedding model...")
    sys.stdout.flush()
    embed_model = get_embedding_model()
    print(f"    Model ready")
    sys.stdout.flush()

    # Embed + upload triples
    if start_idx < len(triples):
        descs = [f"{t['subject']} {t['predicate']} {t['object']}" for t in triples]
        all_embs = embed_texts(descs, embed_model, TRIPLES_EMB_CACHE, "triples")

        print(f"\n  [STAGE] Uploading triples (from idx {start_idx:,})...")
        sys.stdout.flush()
        ok, err = await upload_triples(triples, all_embs, start_idx)
        print(f"    Triples done: {ok:,} uploaded, {err} errors")

    # Upload entities
    if not entities_done:
        await upload_entities(triples, embed_model)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"triples_idx": len(triples), "entities_done": True}, f)

    print(f"\n{'='*60}")
    print(f"  [COMPLETE] Full KG uploaded")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
