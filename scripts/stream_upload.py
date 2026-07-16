#!/usr/bin/env python
"""Stream-upload KG JSONL files to Cosmos DB with 1-letter field remapping.
Reads line-by-line to handle 27GB+ files without loading into memory."""
import asyncio
import json
import os
import random
import sys
import time

from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.identity.aio import AzureCliCredential

COSMOS_URI = "https://divdet-provisioned.documents.azure.com:443/"
TENANT_ID = "43083d15-7273-40c1-b7db-39efd9ccc17a"
DB_NAME = "food"
CONCURRENCY = 200
MAX_RETRIES = 10
BASE_BACKOFF = 0.5

TRIPLES_FILE = "/home/azureuser/AgenticRetrieval-KG/out_kg/kg_triples.jsonl"
ENTITIES_FILE = "/home/azureuser/AgenticRetrieval-KG/out_kg/kg_entities.jsonl"


def remap_triple(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "s": doc.get("subject") or doc.get("pk", ""),
        "p": doc.get("predicate", ""),
        "o": doc.get("object", ""),
        "f": doc.get("confidence", 0.8),
        "n": doc.get("confirmations", 1),
        "d": doc.get("source_chunks", [])[:10],
        "e": doc.get("embedding", []),
    }


def remap_entity(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "n": doc.get("name") or doc.get("pk", ""),
        "t": doc.get("description", ""),
        "r": doc.get("relation_count", 0),
        "d": doc.get("source_chunks", [])[:50],
        "e": doc.get("embedding", []),
    }


async def upload_container(container, filepath, remap_fn, label):
    sem = asyncio.Semaphore(CONCURRENCY)
    uploaded = 0
    errors = 0
    throttles = 0
    t0 = time.time()
    pending = set()

    async def _upsert(doc):
        nonlocal uploaded, errors, throttles
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    await container.upsert_item(doc)
                    uploaded += 1
                    return
                except CosmosHttpResponseError as e:
                    if e.status_code == 429:
                        throttles += 1
                        retry_ms = float(e.headers.get("x-ms-retry-after-ms", 0))
                        wait = max(retry_ms / 1000, BASE_BACKOFF * (2 ** attempt)) + random.uniform(0, 0.3)
                        await asyncio.sleep(wait)
                    else:
                        errors += 1
                        if errors <= 5:
                            print(f"  ERR {e.status_code}: {str(e.message)[:120]}")
                        return
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  ERR: {e}")
                    return
            errors += 1

    line_count = 0
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            remapped = remap_fn(doc)
            line_count += 1

            task = asyncio.create_task(_upsert(remapped))
            pending.add(task)
            task.add_done_callback(pending.discard)

            if len(pending) >= CONCURRENCY * 2:
                done, pending_set = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                pending = pending_set

            if line_count % 5000 == 0:
                elapsed = time.time() - t0
                rate = uploaded / max(elapsed, 1)
                print(f"  [{label}] read={line_count:,} uploaded={uploaded:,} "
                      f"rate={rate:.0f}/s throttles={throttles} errors={errors} "
                      f"elapsed={elapsed:.0f}s")
                sys.stdout.flush()

    if pending:
        await asyncio.gather(*pending)

    elapsed = time.time() - t0
    rate = uploaded / max(elapsed, 1)
    print(f"\n  [{label} DONE] {uploaded:,} uploaded in {elapsed:.0f}s "
          f"({rate:.0f}/s, {errors} errors, {throttles} throttles)")
    sys.stdout.flush()


async def main():
    print("=" * 60)
    print("  Stream Upload KG -> divdet-provisioned (1-letter fields)")
    print("=" * 60)

    cred = AzureCliCredential(tenant_id=TENANT_ID)
    client = CosmosClient(COSMOS_URI, credential=cred)
    db = client.get_database_client(DB_NAME)

    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    if which in ("triples", "both"):
        tc = db.get_container_client("triples")
        print(f"\n>>> Uploading triples from {TRIPLES_FILE}")
        print(f"    Concurrency: {CONCURRENCY}")
        sys.stdout.flush()
        await upload_container(tc, TRIPLES_FILE, remap_triple, "triples")

    if which in ("entities", "both"):
        ec = db.get_container_client("entities")
        print(f"\n>>> Uploading entities from {ENTITIES_FILE}")
        print(f"    Concurrency: {CONCURRENCY}")
        sys.stdout.flush()
        await upload_container(ec, ENTITIES_FILE, remap_entity, "entities")

    await client.close()
    await cred.close()
    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
