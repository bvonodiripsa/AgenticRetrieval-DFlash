#!/usr/bin/env python
"""Parallel upload of Graph Index JSONL to Cosmos DB.

Uses multiprocessing (1 process per shard) + asyncio (concurrent upserts per shard).
Each process gets its own Cosmos client + credential to avoid GIL bottleneck.

Usage:
    python scripts/parallel_upload.py [--workers 8] [--concurrency 40]
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import sys
import time
from functools import partial

import orjson

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def _worker(args):
    """Run in a subprocess — each gets its own event loop + Cosmos client."""
    shard_file, container_name, remap_type, concurrency = args
    tag = os.path.basename(shard_file)

    remap_fn = remap_triple if remap_type == "triple" else remap_entity

    async def _run():
        from azure.cosmos.aio import CosmosClient
        from azure.identity.aio import AzureCliCredential

        cred = AzureCliCredential(tenant_id=TENANT_ID)
        cosmos = CosmosClient(DST_URI, credential=cred)
        ctr = cosmos.get_database_client(DB_NAME).get_container_client(container_name)

        t0 = time.time()
        sem = asyncio.Semaphore(concurrency)
        written = 0
        errors = 0
        total = 0

        async def _upsert(doc):
            nonlocal written, errors
            async with sem:
                for attempt in range(8):
                    try:
                        await ctr.upsert_item(doc)
                        written += 1
                        if written % 2000 == 0:
                            elapsed = time.time() - t0
                            rate = written / max(elapsed, 0.1)
                            print(f"[{tag}] {written:,} written ({rate:.0f}/s, "
                                  f"{elapsed / 60:.1f}m)", flush=True)
                        return
                    except Exception as e:
                        if "429" in str(e) or "TooManyRequests" in str(e):
                            await asyncio.sleep(0.3 * (2 ** attempt))
                        else:
                            errors += 1
                            if errors <= 5:
                                print(f"[{tag}] ERROR: {e}", flush=True)
                            return

        tasks: list[asyncio.Task] = []

        with open(shard_file, "rb") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                total += 1
                doc = orjson.loads(raw_line)
                remapped = remap_fn(doc)
                tasks.append(asyncio.create_task(_upsert(remapped)))

                if len(tasks) >= 300:
                    await asyncio.gather(*tasks)
                    tasks.clear()

        if tasks:
            await asyncio.gather(*tasks)

        elapsed = time.time() - t0
        rate = written / max(elapsed, 0.1)
        result = (f"[{tag}] DONE: {written:,}/{total:,} written, "
                  f"{errors} errors, {elapsed / 60:.1f}m ({rate:.0f}/s)")
        print(result, flush=True)

        await cosmos.close()
        await cred.close()
        return result

    return asyncio.run(_run())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=40,
                        help="async concurrency per worker")
    parser.add_argument("--entities-only", action="store_true")
    parser.add_argument("--triples-only", action="store_true")
    args = parser.parse_args()

    jobs = []
    if not args.entities_only:
        for shard in TRIPLE_SHARDS:
            jobs.append((shard, "triples", "triple", args.concurrency))
    if not args.triples_only:
        jobs.append((ENTITIES_FILE, "entities", "entity", args.concurrency))

    print(f"Launching {len(jobs)} workers (concurrency={args.concurrency} each)...",
          flush=True)
    t0 = time.time()

    with mp.Pool(processes=min(len(jobs), args.workers + 1)) as pool:
        results = pool.map(_worker, jobs)

    for r in results:
        print(r)
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
