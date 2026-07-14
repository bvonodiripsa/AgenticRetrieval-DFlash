#!/usr/bin/env python
"""Single-process, high-concurrency Graph Index uploader.

Reads JSONL files as async generators, fires upserts through a shared
Cosmos client with a single credential and connection pool.

Usage:
    python scripts/turbo_upload.py [--concurrency 300] [--triples-only] [--entities-only]
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

import orjson

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

DST_URI = "https://divdet-provisioned.documents.azure.com:443/"
TENANT_ID = "43083d15-7273-40c1-b7db-39efd9ccc17a"
DB_NAME = "food"

TRIPLE_SHARDS = sorted(
    f"/tmp/gi_shards/{n}" for n in os.listdir("/tmp/gi_shards")
    if n.startswith("triples_")
)
ENTITIES_FILE = "/home/azureuser/AgenticRetrieval-KG/out_kg/kg_entities.jsonl"


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


async def upload_file(ctr, filepath: str, remap_fn, concurrency: int, label: str):
    """Upload a single JSONL file with high async concurrency."""
    t0 = time.time()
    sem = asyncio.Semaphore(concurrency)
    written = 0
    errors = 0
    total = 0
    throttled = 0

    async def _upsert(doc):
        nonlocal written, errors, throttled
        async with sem:
            for attempt in range(12):
                try:
                    await ctr.upsert_item(doc)
                    written += 1
                    return
                except Exception as e:
                    estr = str(e)
                    if "429" in estr or "TooManyRequests" in estr:
                        throttled += 1
                        await asyncio.sleep(0.2 * (2 ** min(attempt, 5)))
                    else:
                        errors += 1
                        if errors <= 5:
                            print(f"  [{label}] ERROR: {e}", flush=True)
                        return

    pending: list[asyncio.Task] = []

    with open(filepath, "rb") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            total += 1
            doc = orjson.loads(raw_line)
            remapped = remap_fn(doc)
            pending.append(asyncio.create_task(_upsert(remapped)))

            if len(pending) >= concurrency * 3:
                await asyncio.gather(*pending)
                pending.clear()
                elapsed = time.time() - t0
                rate = written / max(elapsed, 0.1)
                print(f"  [{label}] {written:,}/{total:,} written "
                      f"({rate:.0f}/s, {elapsed/60:.1f}m, "
                      f"{throttled} throttles)", flush=True)

    if pending:
        await asyncio.gather(*pending)

    elapsed = time.time() - t0
    rate = written / max(elapsed, 0.1)
    print(f"  [{label}] DONE: {written:,}/{total:,} written, "
          f"{errors} errors, {throttled} throttles, "
          f"{elapsed/60:.1f}m ({rate:.0f}/s)", flush=True)
    return written, total, errors


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=300)
    parser.add_argument("--triples-only", action="store_true")
    parser.add_argument("--entities-only", action="store_true")
    args = parser.parse_args()

    cred = AzureCliCredential(tenant_id=TENANT_ID)
    cosmos = CosmosClient(DST_URI, credential=cred)
    db = cosmos.get_database_client(DB_NAME)
    triples_ctr = db.get_container_client("triples")
    entities_ctr = db.get_container_client("entities")

    t0 = time.time()
    grand_written = 0
    grand_total = 0

    if not args.entities_only:
        print(f"\n=== TRIPLES: {len(TRIPLE_SHARDS)} shards, "
              f"concurrency={args.concurrency} ===", flush=True)
        for shard in TRIPLE_SHARDS:
            w, t, _ = await upload_file(
                triples_ctr, shard, remap_triple,
                args.concurrency, os.path.basename(shard))
            grand_written += w
            grand_total += t

    if not args.triples_only:
        print(f"\n=== ENTITIES: concurrency={args.concurrency} ===", flush=True)
        w, t, _ = await upload_file(
            entities_ctr, ENTITIES_FILE, remap_entity,
            args.concurrency, "entities")
        grand_written += w
        grand_total += t

    elapsed = time.time() - t0
    rate = grand_written / max(elapsed, 0.1)
    print(f"\n=== ALL DONE: {grand_written:,}/{grand_total:,} written in "
          f"{elapsed/60:.1f}m ({rate:.0f}/s) ===", flush=True)

    await cosmos.close()
    await cred.close()


if __name__ == "__main__":
    asyncio.run(main())
