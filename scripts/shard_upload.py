#!/usr/bin/env python
"""Upload a single JSONL shard to Cosmos DB. Run multiple instances in parallel.

Usage:
    python scripts/shard_upload.py SHARD_FILE CONTAINER remap_type [--skip N]

    remap_type: "triple" or "entity"
    --skip N: skip first N lines (for resuming)

Example:
    python scripts/shard_upload.py /tmp/gi_shards/triples_aa triples triple &
    python scripts/shard_upload.py /tmp/gi_shards/triples_ab triples triple &
"""

from __future__ import annotations

import asyncio
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

DST_URI = "https://divdet-provisioned.documents.azure.com:443/"
TENANT_ID = "43083d15-7273-40c1-b7db-39efd9ccc17a"
DB_NAME = "food"
CONCURRENCY = 60


def remap_triple(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "s": doc.get("subject", ""),
        "p": doc.get("predicate", ""),
        "o": doc.get("object", ""),
        "f": doc.get("confidence", 0.8),
        "n": doc.get("confirmations", 1),
        "d": doc.get("source_chunks", []),
        "e": doc.get("embedding", []),
    }


def remap_entity(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "n": doc.get("name", ""),
        "t": doc.get("description", ""),
        "r": doc.get("relation_count", 0),
        "d": doc.get("source_chunks", []),
        "e": doc.get("embedding", []),
    }


REMAPPERS = {"triple": remap_triple, "entity": remap_entity}


async def main():
    shard_file = sys.argv[1]
    container_name = sys.argv[2]
    remap_type = sys.argv[3]
    skip = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "--skip" else 0
    if "--skip" in sys.argv:
        idx = sys.argv.index("--skip")
        skip = int(sys.argv[idx + 1])

    remap_fn = REMAPPERS[remap_type]
    tag = os.path.basename(shard_file)

    cred = AzureCliCredential(tenant_id=TENANT_ID)
    cosmos = CosmosClient(DST_URI, credential=cred)
    dst = cosmos.get_database_client(DB_NAME).get_container_client(container_name)

    print(f"[{tag}] Starting: {shard_file} → {container_name} (skip={skip})", flush=True)

    t0 = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)
    written = 0
    errors = 0
    total = 0
    lock = asyncio.Lock()
    pending: list[asyncio.Task] = []

    async def _upsert(remapped):
        nonlocal written, errors
        async with sem:
            for attempt in range(10):
                try:
                    await dst.upsert_item(remapped)
                    async with lock:
                        written += 1
                        if written % 5000 == 0:
                            elapsed = time.time() - t0
                            rate = written / max(elapsed, 0.1)
                            print(f"[{tag}] {written:,} written ({rate:.0f}/s, "
                                  f"{elapsed/60:.1f}m)", flush=True)
                    return
                except Exception as e:
                    if "429" in str(e) or "TooManyRequests" in str(e):
                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        async with lock:
                            errors += 1
                            if errors <= 3:
                                print(f"[{tag}] ERROR: {e}", flush=True)
                        return

    with open(shard_file, "r") as f:
        for line in f:
            total += 1
            if total <= skip:
                continue
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            remapped = remap_fn(doc)

            pending.append(asyncio.create_task(_upsert(remapped)))
            if len(pending) >= 500:
                await asyncio.gather(*pending)
                pending.clear()

    if pending:
        await asyncio.gather(*pending)

    elapsed = time.time() - t0
    rate = written / max(elapsed, 0.1)
    print(f"[{tag}] DONE: {written:,}/{total:,} written, {errors} errors, "
          f"{elapsed/60:.1f}m ({rate:.0f}/s)", flush=True)

    await cosmos.close()
    await cred.close()


if __name__ == "__main__":
    asyncio.run(main())
