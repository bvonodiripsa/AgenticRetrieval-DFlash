#!/usr/bin/env python
"""Fast upload from local JSONL files to provisioned Cosmos DB.

Reads triples/entities from local JSONL (with embeddings), remaps to
1-letter fields, and writes at high concurrency to the provisioned account.

For food products, reads from the serverless source (no local JSONL available).

Usage:
    python scripts/fast_upload.py
    python scripts/fast_upload.py --skip-food   # skip food, only upload Graph Index
"""

from __future__ import annotations

import asyncio
import json
import time
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

SRC_URI = "https://divdet.documents.azure.com:443/"
DST_URI = "https://divdet-provisioned.documents.azure.com:443/"
TENANT_ID = "43083d15-7273-40c1-b7db-39efd9ccc17a"
DB_NAME = "food"

TRIPLES_JSONL = "/home/azureuser/AgenticRetrieval-KG/out_kg/kg_triples.jsonl"
ENTITIES_JSONL = "/home/azureuser/AgenticRetrieval-KG/out_kg/kg_entities.jsonl"

WRITE_CONCURRENCY = 200


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


async def upload_jsonl(dst_db, jsonl_path: str, container_name: str, remap_fn, label: str):
    """Read local JSONL, remap, and write concurrently."""
    dst = dst_db.get_container_client(container_name)

    print(f"\n{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"  Source: {jsonl_path}", flush=True)
    print(f"  Dest:   {container_name} (concurrency={WRITE_CONCURRENCY})", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    sem = asyncio.Semaphore(WRITE_CONCURRENCY)
    written = 0
    errors = 0
    total_lines = 0
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
                            print(f"    {written:,}/{total_lines:,} written ({rate:.0f}/s, "
                                  f"{elapsed/60:.1f}m elapsed)", flush=True)
                    return
                except Exception as e:
                    if "429" in str(e) or "TooManyRequests" in str(e):
                        await asyncio.sleep(0.3 * (attempt + 1))
                    else:
                        async with lock:
                            errors += 1
                            if errors <= 5:
                                print(f"    ERROR: {e}", flush=True)
                        return

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            doc = json.loads(line)
            remapped = remap_fn(doc)

            if total_lines == 1:
                sample = {k: v for k, v in remapped.items() if k != "e"}
                print(f"  Sample: {json.dumps(sample, indent=2, ensure_ascii=False)[:400]}", flush=True)

            if total_lines % 50000 == 0:
                print(f"    Queued {total_lines:,} docs...", flush=True)

            pending.append(asyncio.create_task(_upsert(remapped)))
            if len(pending) >= 2000:
                await asyncio.gather(*pending)
                pending.clear()

    if pending:
        await asyncio.gather(*pending)

    elapsed = time.time() - t0
    rate = written / max(elapsed, 0.1)
    print(f"  DONE: {written:,}/{total_lines:,} written, {errors} errors "
          f"in {elapsed/60:.1f}m ({rate:.0f}/s)", flush=True)
    return written


async def copy_food_from_cosmos(src_db, dst_db):
    """Copy food container from serverless source (no local file)."""
    src = src_db.get_container_client("food")
    dst = dst_db.get_container_client("food")

    print(f"\n{'='*60}", flush=True)
    print(f"  Step 1/3: Food products (from Cosmos source)", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    sem = asyncio.Semaphore(WRITE_CONCURRENCY)
    read_count = 0
    written = 0
    errors = 0
    lock = asyncio.Lock()
    pending: list[asyncio.Task] = []

    async def _upsert(doc):
        nonlocal written, errors
        for k in ("_rid", "_self", "_etag", "_attachments", "_ts"):
            doc.pop(k, None)
        async with sem:
            for attempt in range(10):
                try:
                    await dst.upsert_item(doc)
                    async with lock:
                        written += 1
                        if written % 5000 == 0:
                            elapsed = time.time() - t0
                            rate = written / max(elapsed, 0.1)
                            print(f"    {written:,} written ({rate:.0f}/s)", flush=True)
                    return
                except Exception as e:
                    if "429" in str(e) or "TooManyRequests" in str(e):
                        await asyncio.sleep(0.3 * (attempt + 1))
                    else:
                        async with lock:
                            errors += 1
                            if errors <= 5:
                                print(f"    ERROR: {e}", flush=True)
                        return

    async for item in src.query_items(query="SELECT * FROM c", max_item_count=1000):
        read_count += 1
        if read_count % 10000 == 0:
            print(f"    Read {read_count:,} ({time.time() - t0:.1f}s)", flush=True)

        pending.append(asyncio.create_task(_upsert(item)))
        if len(pending) >= 2000:
            await asyncio.gather(*pending)
            pending.clear()

    if pending:
        await asyncio.gather(*pending)

    elapsed = time.time() - t0
    rate = written / max(elapsed, 0.1)
    print(f"  DONE: {written:,} written, {errors} errors in {elapsed/60:.1f}m ({rate:.0f}/s)", flush=True)
    return written


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-food", action="store_true")
    args = parser.parse_args()

    cred = AzureCliCredential(tenant_id=TENANT_ID)
    dst_cosmos = CosmosClient(DST_URI, credential=cred)
    dst_db = dst_cosmos.get_database_client(DB_NAME)

    print("=" * 60, flush=True)
    print("Fast Upload to divdet-provisioned", flush=True)
    print(f"  Dest: {DST_URI}", flush=True)
    print(f"  Write concurrency: {WRITE_CONCURRENCY}", flush=True)
    print("=" * 60, flush=True)

    t_total = time.time()

    if not args.skip_food:
        src_cosmos = CosmosClient(SRC_URI, credential=cred)
        src_db = src_cosmos.get_database_client(DB_NAME)
        await copy_food_from_cosmos(src_db, dst_db)
        await src_cosmos.close()

    await upload_jsonl(dst_db, TRIPLES_JSONL, "triples", remap_triple,
                       "Step 2/3: Graph Index triples (from local JSONL)")
    await upload_jsonl(dst_db, ENTITIES_JSONL, "entities", remap_entity,
                       "Step 3/3: Graph Index entities (from local JSONL)")

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"All done in {total_elapsed/60:.1f} minutes", flush=True)
    print(f"{'='*60}", flush=True)

    await dst_cosmos.close()
    await cred.close()


if __name__ == "__main__":
    asyncio.run(main())
