#!/usr/bin/env python
"""Copy all data from serverless divdet to provisioned divdet-provisioned.

Copies 3 containers:
  1. food       (58K docs)  — as-is (no field renaming)
  2. kg_triples_food (892K) → triples — remap to 1-letter fields
  3. kg_entities_food (120K) → entities — remap to 1-letter fields

Usage:
    python scripts/migrate_to_provisioned.py
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

SRC_URI = "https://divdet.documents.azure.com:443/"
DST_URI = "https://divdet-provisioned.documents.azure.com:443/"
TENANT_ID = "43083d15-7273-40c1-b7db-39efd9ccc17a"
DB_NAME = "food"
CONCURRENCY = 80


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


def remap_food(doc: dict) -> dict:
    """Copy food docs as-is, just strip Cosmos metadata."""
    clean = {}
    for k, v in doc.items():
        if k.startswith("_") and k not in ("_ts",):
            continue
        clean[k] = v
    clean.pop("_rid", None)
    clean.pop("_self", None)
    clean.pop("_etag", None)
    clean.pop("_attachments", None)
    clean.pop("_ts", None)
    return clean


async def copy_container(
    src_db, dst_db,
    src_name: str, dst_name: str,
    remap_fn,
    label: str,
):
    src = src_db.get_container_client(src_name)
    dst = dst_db.get_container_client(dst_name)

    print(f"\n{'='*60}", flush=True)
    print(f"  {label}: {src_name} → {dst_name}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)
    read_count = 0
    written = 0
    errors = 0
    lock = asyncio.Lock()
    pending: list[asyncio.Task] = []

    async def _upsert(remapped):
        nonlocal written, errors
        async with sem:
            for attempt in range(8):
                try:
                    await dst.upsert_item(remapped)
                    async with lock:
                        written += 1
                        if written % 5000 == 0:
                            elapsed = time.time() - t0
                            rate = written / max(elapsed, 0.1)
                            print(f"    {written:,} written ({rate:.0f}/s)", flush=True)
                    return
                except Exception as e:
                    if "429" in str(e) or "TooManyRequests" in str(e):
                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        async with lock:
                            errors += 1
                            if errors <= 3:
                                print(f"    ERROR: {e}", flush=True)
                        return

    async for item in src.query_items(query="SELECT * FROM c", max_item_count=1000):
        read_count += 1
        remapped = remap_fn(item)

        if read_count == 1:
            sample = {k: v for k, v in remapped.items() if k != "e"}
            print(f"  Sample: {json.dumps(sample, indent=2, ensure_ascii=False)[:500]}", flush=True)

        if read_count % 10000 == 0:
            print(f"    Read {read_count:,} ({time.time() - t0:.1f}s)", flush=True)

        pending.append(asyncio.create_task(_upsert(remapped)))
        if len(pending) >= 1000:
            await asyncio.gather(*pending)
            pending.clear()

    if pending:
        await asyncio.gather(*pending)

    elapsed = time.time() - t0
    rate = written / max(elapsed, 0.1)
    print(f"  DONE: {written:,} written, {errors} errors in {elapsed:.1f}s ({rate:.0f}/s)", flush=True)
    return written


async def main():
    cred = AzureCliCredential(tenant_id=TENANT_ID)
    src_cosmos = CosmosClient(SRC_URI, credential=cred)
    dst_cosmos = CosmosClient(DST_URI, credential=cred)

    src_db = src_cosmos.get_database_client(DB_NAME)
    dst_db = dst_cosmos.get_database_client(DB_NAME)

    print("=" * 60, flush=True)
    print("Full Migration: divdet (serverless) → divdet-provisioned", flush=True)
    print(f"  Source: {SRC_URI}", flush=True)
    print(f"  Dest:   {DST_URI}", flush=True)
    print(f"  Concurrency: {CONCURRENCY}", flush=True)
    print("=" * 60, flush=True)

    t_total = time.time()

    await copy_container(src_db, dst_db, "food", "food", remap_food, "Step 1/3: Food products")
    await copy_container(src_db, dst_db, "kg_triples_food", "triples", remap_triple, "Step 2/3: Graph Index triples")
    await copy_container(src_db, dst_db, "kg_entities_food", "entities", remap_entity, "Step 3/3: Graph Index entities")

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"All done in {total_elapsed/60:.1f} minutes", flush=True)
    print(f"{'='*60}", flush=True)

    await src_cosmos.close()
    await dst_cosmos.close()
    await cred.close()


if __name__ == "__main__":
    asyncio.run(main())
